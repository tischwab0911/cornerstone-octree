/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

 /*! @file
 * @brief Generic octree traversal methods
 *
 * @author Timo Schwab <tischwab@ethz.ch>
 *
 * Dual tree traversal on the GPU is the basis for many algorithms.
 *
 * Architecture overview
 * ─────────────────────
 * Three levels of work distribution:
 *   GPU  → clusters : assignPairByBalancedSplit splits the root pair across clusters
 *   TBC  → blocks   : assignPairByBalancedSplit splits again across blocks in a cluster
 *   Block→ warps    : per-warp DFS stacks + shared queue for load balancing
 *
 * Block-level design (synchronized two-phase model):
 *   Two phases separated by block.sync():
 *     Phase 1 (PRODUCE): All warps traverse. Per-warp DFS stacks expand
 *       the frontier; leaf pairs append to flat p2p/m2l buffers via single
 *       atomicAdd (no ring buffer, no committed pointer, no backpressure).
 *     Phase 2 (DRAIN): All warps consume. Each warp is assigned to either
 *       p2p or m2l (not both) proportional to buffer fill — zero warp divergence.
 *   2 block.sync() per iteration. No role management, no busy flags, no
 *   volatile polling.
 *
 * Shared traversal queue (for inter-warp load balancing):
 *   tail      – reservation cursor, advanced by atomicAdd before writing data
 *   committed – advanced by atomicAdd after writing data + __threadfence_block()
 *   head      – advanced by atomicCAS when dequeuing
 *   Invariant: head <= committed <= tail  (modulo wrap)
 */

#pragma once
#pragma nv_diag_suppress static_var_with_dynamic_init

#include "cstone/tree/octree.hpp"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/warpscan.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>

namespace cg = cooperative_groups;

__device__ __forceinline__ unsigned linear_block_rank()
{
    cg::grid_group grid = cg::this_grid();
    return grid.block_rank();
}

__device__ __forceinline__ unsigned blocks_per_cluster_runtime()
{
    cg::cluster_group cl = cg::this_cluster();
    return cl.num_blocks();
}

__device__ __forceinline__ unsigned block_rank_in_cluster()
{
    cg::cluster_group cl = cg::this_cluster();
    return cl.block_rank();
}

__device__ __forceinline__ unsigned cluster_rank_in_grid()
{
    return linear_block_rank() / blocks_per_cluster_runtime();
}

__device__ __forceinline__ unsigned num_clusters_runtime()
{
    return cg::this_grid().num_blocks() / blocks_per_cluster_runtime();
}

__device__ __forceinline__ unsigned thread_rank_in_block()
{
    return cg::this_thread_block().thread_rank();
}

__device__ __forceinline__ unsigned lane_id()
{
    unsigned r;
    asm("mov.u32 %0, %laneid;" : "=r"(r));
    return r;
}

HOST_DEVICE_FUN __forceinline__
constexpr std::size_t align_up(std::size_t x, std::size_t a)
{
    return (x + (a - 1)) & ~(a - 1);
}

namespace cstone
{

__device__ __forceinline__ bool isLeaf(const TreeNodeIndex* __restrict__ childOffsets,
                                       TreeNodeIndex n)
{
    return childOffsets[n] == 0;
}

__device__ __forceinline__ unsigned warp_id_in_block()
{
    return threadIdx.x >> 5;
}

// ──────────────────────────────────────────────────────────────────────────────
// Work splitting: assign a sub-pair to each unit (cluster or block) by
// recursively subdividing the larger node of the starting pair.
// ──────────────────────────────────────────────────────────────────────────────

template<class MAC>
__device__ __forceinline__
bool assignPairByBalancedSplit(const TreeNodeIndex* __restrict__ childOffsets,
                               TreeNodeIndex& a, TreeNodeIndex& b,
                               unsigned rid, unsigned N,
                               unsigned& active_count,
                               MAC&& continuation)
{
    if (N <= 1) { active_count = 1; return rid == 0; }

    while (N > 1)
    {
        bool canSplitA = !isLeaf(childOffsets, a);
        bool canSplitB = !isLeaf(childOffsets, b);

        if ((!canSplitA && !canSplitB) || !continuation(a, b))
        {
            // Can't split further: only one unit processes this pair
            active_count = 1;
            return rid == 0;
        }

        // Split the node closer to root (smaller index = higher in tree)
        bool splitA;
        if      (!canSplitA) splitA = false;
        else if (!canSplitB) splitA = true;
        else                 splitA = (a <= b);

        TreeNodeIndex splitNode = splitA ? a : b;
        unsigned childBase = childOffsets[splitNode];

        // Balanced distribution: N units among 8 children
        unsigned perChild = N >> 3;       // N / 8
        unsigned extra    = N & 7u;       // N % 8
        // First `extra` children get (perChild+1) units, rest get perChild
        unsigned threshold = extra * (perChild + 1);

        unsigned child_idx, local_N;
        if (perChild == 0) {
            // N < 8: each of the first N children gets exactly 1 unit
            child_idx = rid;
            rid = 0;
            local_N = 1;
        } else if (rid < threshold) {
            child_idx = rid / (perChild + 1);
            rid       = rid % (perChild + 1);
            local_N   = perChild + 1;
        } else {
            unsigned adjusted = rid - threshold;
            child_idx = extra + adjusted / perChild;
            rid       = adjusted % perChild;
            local_N   = perChild;
        }

        if (splitA) a = childBase + child_idx;
        else        b = childBase + child_idx;

        N = local_N;
    }

    active_count = 1;
    return rid == 0;
}

// ──────────────────────────────────────────────────────────────────────────────
// Shared memory layout
//
// Per-warp DFS stacks:
//   warpStackA[numWarps * stackCap]   — target node indices
//   warpStackB[numWarps * stackCap]   — source node indices
//   stackTops[numWarps]               — current depth of each warp's stack
//
// Shared traversal queue (ring buffer, for inter-warp load balancing):
//   sharedA[sharedCap], sharedB[sharedCap]
//   QueuePtrs (head/tail/committed)
//
// ProducerDesc pool:
//   descs[numWarps * 32]              — warp expansion descriptors
//
// Flat p2p/m2l buffers (drained each iteration):
//   p2pA[leafBufCap], p2pB[leafBufCap], p2pCount
//   m2lA[leafBufCap], m2lB[leafBufCap], m2lCount
// ──────────────────────────────────────────────────────────────────────────────

struct QueuePtrs
{
    unsigned head;
    unsigned tail;       // reservation cursor
    unsigned committed;  // data-ready cursor
};

struct SmemLayout
{
    unsigned stackCap;     // per-warp stack capacity
    unsigned sharedCap;    // shared trav queue capacity
    unsigned leafBufCap;   // flat p2p/m2l buffer capacity
    unsigned offWarpStackA;
    unsigned offWarpStackB;
    unsigned offStackTops;
    unsigned offSharedA;
    unsigned offSharedB;
    unsigned offSharedPtrs;
    unsigned offDesc;
    unsigned offP2pA;
    unsigned offP2pB;
    unsigned offP2pCount;
    unsigned offM2lA;
    unsigned offM2lB;
    unsigned offM2lCount;
    unsigned offDone;
};

HOST_DEVICE_FUN __forceinline__
SmemLayout computeLayout(unsigned stackCap, unsigned sharedCap, unsigned leafBufCap, unsigned numWarps)
{
    SmemLayout L;
    L.stackCap   = stackCap;
    L.sharedCap  = sharedCap;
    L.leafBufCap = leafBufCap;

    unsigned cur = 0;

    // Per-warp stacks (A and B)
    L.offWarpStackA = cur;
    cur += numWarps * stackCap * sizeof(TreeNodeIndex);
    L.offWarpStackB = cur;
    cur += numWarps * stackCap * sizeof(TreeNodeIndex);

    // Stack tops
    L.offStackTops = cur;
    cur += (unsigned)align_up(numWarps * sizeof(unsigned), 4);

    // Shared traversal queue
    L.offSharedA = cur;
    cur += sharedCap * sizeof(TreeNodeIndex);
    L.offSharedB = cur;
    cur += sharedCap * sizeof(TreeNodeIndex);
    L.offSharedPtrs = cur;
    cur += (unsigned)align_up(sizeof(QueuePtrs), 4);

    // ProducerDesc pool (12 bytes each, numWarps * 32 entries)
    L.offDesc = cur;
    cur += numWarps * 32u * 12u;

    // P2P flat buffer
    L.offP2pA = cur;
    cur += leafBufCap * sizeof(TreeNodeIndex);
    L.offP2pB = cur;
    cur += leafBufCap * sizeof(TreeNodeIndex);
    L.offP2pCount = cur;
    cur += (unsigned)align_up(sizeof(unsigned), 4);

    // M2L flat buffer
    L.offM2lA = cur;
    cur += leafBufCap * sizeof(TreeNodeIndex);
    L.offM2lB = cur;
    cur += leafBufCap * sizeof(TreeNodeIndex);
    L.offM2lCount = cur;
    cur += (unsigned)align_up(sizeof(unsigned), 4);

    // Done flag for termination
    L.offDone = cur;

    return L;
}

/*! @brief Compute the required dynamic shared memory bytes.
 *
 * @param queueCap   capacity parameter: per-warp stack = queueCap/numWarps, shared trav queue = queueCap
 * @param numWarps   number of warps per block
 */
HOST_DEVICE_FUN inline unsigned dualTraversalSmemBytes(unsigned queueCap, unsigned numWarps)
{
    constexpr unsigned kMaxPop = 4;
    unsigned stackCap   = queueCap / numWarps;
    unsigned sharedCap  = queueCap;
    unsigned leafBufCap = numWarps * kMaxPop * 12;

    unsigned stackBytes   = numWarps * stackCap * 2u * (unsigned)sizeof(TreeNodeIndex);
    unsigned topBytes     = (unsigned)align_up(numWarps * sizeof(unsigned), 4);
    unsigned travBytes    = sharedCap * 2u * (unsigned)sizeof(TreeNodeIndex);
    unsigned travPtrBytes = (unsigned)align_up(sizeof(QueuePtrs), 4);
    unsigned descBytes    = numWarps * 32u * 12u;
    unsigned p2pBytes     = leafBufCap * 2u * (unsigned)sizeof(TreeNodeIndex) + (unsigned)align_up(sizeof(unsigned), 4);
    unsigned m2lBytes     = leafBufCap * 2u * (unsigned)sizeof(TreeNodeIndex) + (unsigned)align_up(sizeof(unsigned), 4);
    unsigned doneBytes    = (unsigned)align_up(sizeof(unsigned), 4);

    unsigned total = stackBytes + topBytes + travBytes + travPtrBytes + descBytes
                   + p2pBytes + m2lBytes + doneBytes;
    return (unsigned)align_up(total, 128);
}

__device__ __forceinline__ TreeNodeIndex* smemArr(char* s, unsigned off)
{
    return reinterpret_cast<TreeNodeIndex*>(s + off);
}
__device__ __forceinline__ QueuePtrs* smemPtrs(char* s, unsigned off)
{
    return reinterpret_cast<QueuePtrs*>(s + off);
}

// ──────────────────────────────────────────────────────────────────────────────
// Warp-cooperative push to the shared load-balancing queue (ring buffer)
// ──────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__
void warp_push_pairs(unsigned pushMask,
                     QueuePtrs* __restrict__ ptrs,
                     TreeNodeIndex* __restrict__ A,
                     TreeNodeIndex* __restrict__ B,
                     unsigned cap,
                     TreeNodeIndex a, TreeNodeIndex b)
{
    int n = __popc(pushMask);
    if (n == 0) return;

    unsigned lane = lane_id();
    int inGroup = (pushMask >> lane) & 1u;
    int rank = __popc(pushMask & ((1u << lane) - 1u));

    // Reserve slots in the ring buffer.  If the queue is nearly full,
    // spin-wait for consumers to drain it (backpressure).
    unsigned base;
    unsigned leader = __ffs(pushMask) - 1;
    if (lane == leader) {
        base = atomicAdd(&ptrs->tail, (unsigned)n);
        // Wait until all our reserved slots are safe to write
        // (i.e. consumers have advanced head past the wrap-around point)
        while (base + (unsigned)n - atomicAdd(&ptrs->head, 0u) > cap) {}
    }
    base = __shfl_sync(0xFFFFFFFF, base, leader);

    if (inGroup) {
        unsigned pos = (base + (unsigned)rank) % cap;
        A[pos] = a;
        B[pos] = b;
    }

    __threadfence_block();
    if (lane == leader) {
        atomicAdd(&ptrs->committed, (unsigned)n);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Warp-cooperative push to flat p2p/m2l buffers (no ring buffer, no fence)
// ──────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__
void warp_push_flat(unsigned pushMask,
                    TreeNodeIndex* __restrict__ A,
                    TreeNodeIndex* __restrict__ B,
                    unsigned* __restrict__ count,
                    TreeNodeIndex a, TreeNodeIndex b)
{
    int n = __popc(pushMask);
    if (n == 0) return;

    unsigned lane = lane_id();
    int rank = __popc(pushMask & ((1u << lane) - 1u));

    unsigned base;
    unsigned leader = __ffs(pushMask) - 1;
    if (lane == leader)
        base = atomicAdd(count, (unsigned)n);
    base = __shfl_sync(0xFFFFFFFF, base, leader);

    if ((pushMask >> lane) & 1u) {
        A[base + rank] = a;
        B[base + rank] = b;
    }
    // No __threadfence_block(), no committed pointer.
    // block.sync() at Phase 1 end handles visibility.
}

// ──────────────────────────────────────────────────────────────────────────────
// Warp-cooperative dequeue from the shared queue
// ──────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__
bool warpDequeue(TreeNodeIndex* __restrict__ A,
                 TreeNodeIndex* __restrict__ B,
                 QueuePtrs* __restrict__     ptrs,
                 unsigned                    cap,
                 unsigned                    maxItems,
                 TreeNodeIndex&              outA,
                 TreeNodeIndex&              outB)
{
    const unsigned laneIdx = lane_id();

    unsigned slotBase = 0;
    unsigned count    = 0;

    if (laneIdx == 0) {
        unsigned oldHead = atomicAdd(&ptrs->head, 0u);

        for (;;) {
            unsigned curCommitted = atomicAdd(&ptrs->committed, 0u);
            unsigned available = curCommitted - oldHead;

            if (available == 0) {
                count = 0;
                break;
            }

            count = min(available, maxItems);

            unsigned prev = atomicCAS(&ptrs->head, oldHead, oldHead + count);
            if (prev == oldHead) {
                slotBase = oldHead;
                break;
            }
            oldHead = prev;
        }
    }

    count    = __shfl_sync(0xFFFFFFFF, count,    0);
    slotBase = __shfl_sync(0xFFFFFFFF, slotBase, 0);

    if (count == 0) return false;

    __threadfence_block();

    bool active = laneIdx < count;
    if (active) {
        unsigned idx = (slotBase + laneIdx) % cap;
        outA = A[idx];
        outB = B[idx];
    }

    return active;
}

// ──────────────────────────────────────────────────────────────────────────────
// Warp-cooperative push: local stack first, overflow to shared queue
// ──────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__
void warp_push_stack_or_queue(
    unsigned pushMask,
    TreeNodeIndex* __restrict__ stackA,
    TreeNodeIndex* __restrict__ stackB,
    unsigned* __restrict__      stackTop,
    unsigned                    stackCap,
    QueuePtrs* __restrict__     sharedPtrs,
    TreeNodeIndex* __restrict__ sharedA,
    TreeNodeIndex* __restrict__ sharedB,
    unsigned                    sharedCap,
    TreeNodeIndex a, TreeNodeIndex b)
{
    int n = __popc(pushMask);
    if (n == 0) return;

    unsigned lane = lane_id();
    bool inGroup = (pushMask >> lane) & 1u;
    int rank = inGroup ? __popc(pushMask & ((1u << lane) - 1u)) : -1;

    // All lanes read stack top (uniform within warp — only this warp writes it)
    unsigned top   = *stackTop;
    unsigned space = stackCap > top ? stackCap - top : 0;
    unsigned toStack = min((unsigned)n, space);

    // First `toStack` items (by rank) go to the local stack
    if (inGroup && (unsigned)rank < toStack) {
        stackA[top + rank] = a;
        stackB[top + rank] = b;
    }

    if (lane == 0) *stackTop = top + toStack;
    __syncwarp(0xFFFFFFFF);

    // Remaining items overflow to the shared queue
    if (toStack < (unsigned)n) {
        unsigned overflowBit = (inGroup && (unsigned)rank >= toStack) ? 1u : 0u;
        unsigned overflowMask = __ballot_sync(0xFFFFFFFF, overflowBit);
        warp_push_pairs(overflowMask, sharedPtrs, sharedA, sharedB, sharedCap, a, b);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Warp steal: pull items from the shared queue into the local stack
// ──────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__
void warp_steal_to_stack(
    TreeNodeIndex* __restrict__ stackA,
    TreeNodeIndex* __restrict__ stackB,
    unsigned* __restrict__      stackTop,
    unsigned                    stackCap,
    TreeNodeIndex* __restrict__ sharedA,
    TreeNodeIndex* __restrict__ sharedB,
    QueuePtrs* __restrict__     sharedPtrs,
    unsigned                    sharedCap)
{
    unsigned top = *stackTop;
    if (top > 0) return;  // still have local work

    unsigned maxSteal = min(32u, stackCap);
    TreeNodeIndex stealA, stealB;
    bool got = warpDequeue(sharedA, sharedB, sharedPtrs, sharedCap, maxSteal, stealA, stealB);

    unsigned gotMask = __ballot_sync(0xFFFFFFFF, got);
    unsigned nStolen = __popc(gotMask);

    if (got) {
        // warpDequeue gives consecutive lanes items: lane 0 → item 0, etc.
        stackA[lane_id()] = stealA;
        stackB[lane_id()] = stealB;
    }

    if (lane_id() == 0) *stackTop = nStolen;
    __syncwarp(0xFFFFFFFF);
}

// ──────────────────────────────────────────────────────────────────────────────
// Producer expansion: one warp processes dequeued items.
// - non-leaf pairs that pass criterion → push to local stack (overflow to trav queue)
// - leaf-leaf pairs that pass criterion → push to p2p queue
// - pairs that fail criterion          → push to m2l queue
// ──────────────────────────────────────────────────────────────────────────────

struct ProducerDesc
{
    int childBase;
    int fixed;
    unsigned char mode; // 0=subdivide target, 1=subdivide source
};

template<class ContinuationFn>
__device__ __forceinline__
void warpExpand(const TreeNodeIndex* __restrict__ childOffsets,
                ContinuationFn continuation,
                TreeNodeIndex target,
                TreeNodeIndex source,
                bool validItem,
                ProducerDesc* __restrict__ warpDescs,
                unsigned                    stackCap,
                unsigned                    sharedCap,
                TreeNodeIndex* __restrict__ stackA,
                TreeNodeIndex* __restrict__ stackB,
                unsigned* __restrict__      stackTop,
                TreeNodeIndex* __restrict__ sharedA,
                TreeNodeIndex* __restrict__ sharedB,
                QueuePtrs* __restrict__     sharedP,
                TreeNodeIndex* __restrict__ p2pA,
                TreeNodeIndex* __restrict__ p2pB,
                unsigned* __restrict__      p2pCount,
                TreeNodeIndex* __restrict__ m2lA,
                TreeNodeIndex* __restrict__ m2lB,
                unsigned* __restrict__      m2lCount)
{
    const unsigned lane = lane_id();

    // Per-lane: decide whether this item needs 8-way expansion
    bool produce8 = false;
    bool leafLeaf = false;
    ProducerDesc my{0, 0, 0};

    if (validItem)
    {
        const bool targetLeaf = isLeaf(childOffsets, target);
        const bool sourceLeaf = isLeaf(childOffsets, source);

        if ((target < source && !targetLeaf) || sourceLeaf)
        {
            produce8 = !targetLeaf;
            if (produce8) {
                my.mode      = 0;
                my.childBase = childOffsets[target];
                my.fixed     = source;
            }
        }
        else
        {
            produce8 = !sourceLeaf;
            if (produce8) {
                my.mode      = 1;
                my.childBase = childOffsets[source];
                my.fixed     = target;
            }
        }

        // Both leaves: classify as p2p or m2l and push to queue
        if (!produce8) leafLeaf = true;
    }

    // Push leaf-leaf pairs to p2p or m2l queues
    bool leafP2p = leafLeaf && continuation(target, source);
    bool leafM2l = leafLeaf && !leafP2p;

    unsigned mLeafP2p = __ballot_sync(0xFFFFFFFF, leafP2p);
    warp_push_flat(mLeafP2p, p2pA, p2pB, p2pCount, target, source);

    unsigned mLeafM2l = __ballot_sync(0xFFFFFFFF, leafM2l);
    warp_push_flat(mLeafM2l, m2lA, m2lB, m2lCount, target, source);

    // Compact producers within the warp
    unsigned prodMask = __ballot_sync(0xFFFFFFFF, produce8);
    int nProducers    = __popc(prodMask);

    if (produce8) {
        int rank = __popc(prodMask & ((1u << lane) - 1u));
        warpDescs[rank] = my;
    }

    __syncwarp(0xFFFFFFFF);

    // Process all 8*nProducers child pairs
    int total = nProducers * 8;
    int totalRounded = ((total + 31) / 32) * 32;

    for (int j = (int)lane; j < totalRounded; j += 32)
    {
        TreeNodeIndex a = 0, b = 0;
        int dest = 0;  // 0=none, 1=trav, 2=p2p, 3=m2l

        if (j < total)
        {
            int pi  = j >> 3;
            int oct = j & 7;

            ProducerDesc d = warpDescs[pi];

            if (d.mode == 0) { a = d.childBase + oct; b = d.fixed; }
            else             { a = d.fixed; b = d.childBase + oct; }

            if (continuation(a, b))
            {
                bool la = isLeaf(childOffsets, a);
                bool lb = isLeaf(childOffsets, b);
                dest = (la && lb) ? 2 : 1;
            }
            else
            {
                dest = 3;
            }
        }

        // Push traversal items to local stack (overflow to shared queue)
        unsigned mTrav = __ballot_sync(0xFFFFFFFF, dest == 1);
        warp_push_stack_or_queue(mTrav,
                                 stackA, stackB, stackTop, stackCap,
                                 sharedP, sharedA, sharedB, sharedCap,
                                 a, b);

        // Push p2p and m2l pairs to their respective flat buffers
        unsigned mP2P = __ballot_sync(0xFFFFFFFF, dest == 2);
        warp_push_flat(mP2P, p2pA, p2pB, p2pCount, a, b);

        unsigned mM2L = __ballot_sync(0xFFFFFFFF, dest == 3);
        warp_push_flat(mM2L, m2lA, m2lB, m2lCount, a, b);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Block-level dual traversal (synchronized two-phase model)
//
// Two phases separated by block.sync():
//   Phase 1 (PRODUCE): All warps traverse. Leaf pairs append to flat buffers
//                       (single atomicAdd per warp, no committed/fence/backpressure).
//   Phase 2 (DRAIN):   All warps consume. Each warp assigned to either p2p or
//                       m2l (not both) — zero warp divergence.
//
// The shared traversal queue (for inter-warp load balancing) is kept as-is.
// ──────────────────────────────────────────────────────────────────────────────

template<int numWarps, unsigned queueCap, class MAC, class M2L, class P2P>
__device__ void dualTraversalBlock(
    const TreeNodeIndex* __restrict__ childOffsets,
    TreeNodeIndex a, TreeNodeIndex b,
    MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    extern __shared__ char dynSmem[];

    cg::thread_block block = cg::this_thread_block();
    unsigned tid     = block.thread_rank();
    unsigned warpId  = tid / GpuConfig::warpSize;
    unsigned laneIdx = tid % GpuConfig::warpSize;

    // Handle trivial case: both are leaves
    if (isLeaf(childOffsets, a) && isLeaf(childOffsets, b)) {
        if (tid == 0) {
            if (continuation(a, b)) { p2p(a, b); }
            else                    { m2l(a, b); }
        }
        return;
    }

    constexpr unsigned kMaxPop    = 4;
    constexpr unsigned stackCap   = queueCap / numWarps;
    constexpr unsigned sharedCap  = queueCap;
    constexpr unsigned leafBufCap = numWarps * kMaxPop * 12;
    SmemLayout L = computeLayout(stackCap, sharedCap, leafBufCap, numWarps);

    // Per-warp stack pointers
    TreeNodeIndex* myStackA = smemArr(dynSmem, L.offWarpStackA + warpId * stackCap * sizeof(TreeNodeIndex));
    TreeNodeIndex* myStackB = smemArr(dynSmem, L.offWarpStackB + warpId * stackCap * sizeof(TreeNodeIndex));
    unsigned* stackTops     = reinterpret_cast<unsigned*>(dynSmem + L.offStackTops);
    unsigned* myStackTop    = &stackTops[warpId];

    // Shared traversal queue
    TreeNodeIndex* sharedA = smemArr(dynSmem, L.offSharedA);
    TreeNodeIndex* sharedB = smemArr(dynSmem, L.offSharedB);
    QueuePtrs*     sharedP = smemPtrs(dynSmem, L.offSharedPtrs);

    // P2P/M2L flat buffers
    TreeNodeIndex* p2pA     = smemArr(dynSmem, L.offP2pA);
    TreeNodeIndex* p2pB     = smemArr(dynSmem, L.offP2pB);
    unsigned*      p2pCount = reinterpret_cast<unsigned*>(dynSmem + L.offP2pCount);
    TreeNodeIndex* m2lA     = smemArr(dynSmem, L.offM2lA);
    TreeNodeIndex* m2lB     = smemArr(dynSmem, L.offM2lB);
    unsigned*      m2lCount = reinterpret_cast<unsigned*>(dynSmem + L.offM2lCount);

    // ProducerDesc pool (per-warp)
    ProducerDesc* myDescs = reinterpret_cast<ProducerDesc*>(dynSmem + L.offDesc) + warpId * 32;

    // ── Initialization ──
    if (tid == 0) {
        sharedA[0] = a;
        sharedB[0] = b;
        sharedP->head = 0;
        sharedP->tail = 1;
        sharedP->committed = 1;

        *p2pCount = 0;
        *m2lCount = 0;
    }
    for (unsigned w = tid; w < (unsigned)numWarps; w += blockDim.x) {
        stackTops[w] = 0;
    }
    block.sync();

    while (true)
    {
        // ══ PHASE 1: ALL WARPS PRODUCE ══

        // Steal from shared trav queue if local stack is empty
        warp_steal_to_stack(myStackA, myStackB, myStackTop, stackCap,
                            sharedA, sharedB, sharedP, sharedCap);
        __syncwarp(0xFFFFFFFF);

        unsigned top = *myStackTop;
        unsigned nPop = min(kMaxPop, top);

        TreeNodeIndex nodeA = 0, nodeB = 0;
        bool gotItem = false;

        if (laneIdx < nPop) {
            unsigned idx = top - nPop + laneIdx;
            nodeA = myStackA[idx];
            nodeB = myStackB[idx];
            gotItem = true;
        }
        if (laneIdx == 0) *myStackTop = top - nPop;
        __syncwarp(0xFFFFFFFF);

        if (nPop > 0)
            warpExpand(childOffsets, continuation,
                       nodeA, nodeB, gotItem, myDescs,
                       stackCap, sharedCap,
                       myStackA, myStackB, myStackTop,
                       sharedA, sharedB, sharedP,
                       p2pA, p2pB, p2pCount,
                       m2lA, m2lB, m2lCount);

        block.sync();  // ── BARRIER 1: produce done ──

        // ══ TERMINATION CHECK ══
        unsigned nP2p = *p2pCount;
        unsigned nM2l = *m2lCount;

        bool travDone = true;
        for (int w = 0; w < numWarps; ++w)
            if (stackTops[w] != 0) { travDone = false; break; }
        if (travDone && (sharedP->committed != sharedP->head))
            travDone = false;

        if (travDone && nP2p == 0 && nM2l == 0) break;

        // ══ PHASE 2: ALL WARPS DRAIN ══
        if (nP2p > 0 || nM2l > 0)
        {
            // Assign warps proportionally (no divergence within a warp)
            unsigned p2pWarps;
            if      (nM2l == 0) p2pWarps = numWarps;
            else if (nP2p == 0) p2pWarps = 0;
            else {
                p2pWarps = max(1u, min((unsigned)(numWarps - 1),
                    (nP2p * numWarps + (nP2p + nM2l) / 2) / (nP2p + nM2l)));
            }

            if (warpId < p2pWarps) {
                for (unsigned i = warpId * 32 + laneIdx; i < nP2p; i += p2pWarps * 32)
                    p2p(p2pA[i], p2pB[i]);
            } else {
                unsigned mw = warpId - p2pWarps;
                unsigned mWarps = numWarps - p2pWarps;
                for (unsigned i = mw * 32 + laneIdx; i < nM2l; i += mWarps * 32)
                    m2l(m2lA[i], m2lB[i]);
            }
        }

        // Reset counts for next iteration
        if (tid == 0) { *p2pCount = 0; *m2lCount = 0; }

        block.sync();  // ── BARRIER 2: drain done, counts reset ──
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TBC level: split work across blocks within a cluster
// ──────────────────────────────────────────────────────────────────────────────

template <int numWarps, unsigned queueCap, class MAC, class M2L, class P2P>
__device__ void dualTraversalTBC(const TreeNodeIndex* __restrict__ childOffsets,
                                 TreeNodeIndex a, TreeNodeIndex b,
                                 MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    const unsigned block_in_cluster = block_rank_in_cluster();
    const unsigned blocksPerCluster = blocks_per_cluster_runtime();

    unsigned active_blocks = 0;
    const bool active_block =
        assignPairByBalancedSplit(childOffsets, a, b,
                                      block_in_cluster, blocksPerCluster,
                                      active_blocks,
                                      std::forward<MAC>(continuation));

    if (!active_block) return;

    dualTraversalBlock<numWarps, queueCap>(childOffsets, a, b,
                                            std::forward<MAC>(continuation),
                                            std::forward<M2L>(m2l),
                                            std::forward<P2P>(p2p));
}

// ──────────────────────────────────────────────────────────────────────────────
// GPU level: split work across clusters
// ──────────────────────────────────────────────────────────────────────────────

template<int numWarps, unsigned queueCap, class MAC, class M2L, class P2P>
__device__ void dualTraversalGPU(const TreeNodeIndex* __restrict__ childOffsets,
                                 TreeNodeIndex rootA, TreeNodeIndex rootB,
                                 MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    const unsigned cluster_id  = cluster_rank_in_grid();
    const unsigned numClusters = num_clusters_runtime();

    TreeNodeIndex a = rootA, b = rootB;

    unsigned active_clusters = 0;
    const bool active_cluster =
        assignPairByBalancedSplit(childOffsets, a, b,
                                      cluster_id, numClusters,
                                      active_clusters,
                                      std::forward<MAC>(continuation));

    if (!active_cluster) return;

    dualTraversalTBC<numWarps, queueCap>(childOffsets, a, b,
                                          std::forward<MAC>(continuation),
                                          std::forward<M2L>(m2l),
                                          std::forward<P2P>(p2p));
}

} // namespace cstone
