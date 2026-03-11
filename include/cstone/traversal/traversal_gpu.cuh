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
 */

#pragma once
#pragma nv_diag_suppress static_var_with_dynamic_init

#include "cstone/tree/octree.hpp"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/warpscan.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cassert>
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

template<class Kernel>
unsigned maxConcurrentBlocks(Kernel kernel,
                                   unsigned threadsPerBlock,
                                   unsigned blocksPerCluster,
                                   size_t dynamicSmemBytes = 0)
{
    cudaDeviceProp prop{};
    checkGpuErrors(cudaGetDeviceProperties(&prop, 0));

    cudaFuncAttributes attr{};
    checkGpuErrors(cudaFuncGetAttributes(&attr, (const void*)kernel));

    const unsigned regsPerBlock =
        static_cast<unsigned>(attr.numRegs) * threadsPerBlock;

    const size_t smemPerBlock =
        static_cast<size_t>(attr.sharedSizeBytes) + dynamicSmemBytes;

    auto div_floor = [](unsigned a, unsigned b) -> unsigned
    {
        return b ? (a / b) : 0;
    };

    unsigned byBlocks  = static_cast<unsigned>(prop.maxBlocksPerMultiProcessor);
    unsigned byThreads = div_floor(static_cast<unsigned>(prop.maxThreadsPerMultiProcessor),
                                   threadsPerBlock);

    unsigned byRegs = byBlocks;
    if (regsPerBlock)
    {
        byRegs = div_floor(static_cast<unsigned>(prop.regsPerMultiprocessor),
                           regsPerBlock);
    }

    unsigned bySmem = byBlocks;
    if (smemPerBlock)
    {
        bySmem = static_cast<unsigned>(
            prop.sharedMemPerMultiprocessor / smemPerBlock);
    }

    unsigned blocksPerSM = std::min({byBlocks, byThreads, byRegs, bySmem});
    unsigned totalBlocks = blocksPerSM * static_cast<unsigned>(prop.multiProcessorCount);

    // Cluster launches need a multiple of blocksPerCluster.
    if (blocksPerCluster > 1)
    {
        totalBlocks = (totalBlocks / blocksPerCluster) * blocksPerCluster;
    }

    // Keep at least one cluster if the raw estimate says >0 but rounding erased it.
    if (totalBlocks == 0 && blocksPerSM > 0)
    {
        totalBlocks = blocksPerCluster;
    }

    return totalBlocks-192;
}

/*! @brief Try to acquire a shared-memory spinlock flag (0=free, 1=held).
 *  @return 1 if acquired, 0 if contention (non-blocking). Lane 0 only. */
__device__ __forceinline__ int acquireFlag(unsigned* flag)
{
    return atomicCAS(flag, 0u, 1u) == 0u ? 1 : 0;
}

/*! @brief Release a previously acquired flag. Lane 0 only. */
__device__ __forceinline__ void releaseFlag(unsigned* flag)
{
    __threadfence_block();
    atomicExch(flag, 0u);
}


template<unsigned cap>
__device__ __forceinline__ bool ibStackPush(
    TreeNodeIndex* __restrict__ ibA,
    TreeNodeIndex* __restrict__ ibB,
    int*           __restrict__ ibIsP2P,
    unsigned*      __restrict__ ibCount,
    unsigned*      __restrict__ ibFlag,
    const TreeNodeIndex* __restrict__ srcA,
    const TreeNodeIndex* __restrict__ srcB,
    const int*           __restrict__ srcIsP2P,
    unsigned numItems)
{
    auto tile = cg::coalesced_threads();
    unsigned lane = tile.thread_rank();

    int success;
    if (lane == 0) success = acquireFlag(ibFlag);
    success = tile.shfl(success, 0);
    if (!success) return false;

    unsigned base;
    if (lane == 0) base = *ibCount;
    base = tile.shfl(base, 0);

    if (base + numItems > cap)
    {
        if (lane == 0) releaseFlag(ibFlag);
        return false;
    }

    for (unsigned i = lane; i < numItems; i += GpuConfig::warpSize)
    {
        // if (base + i >= cap) assert(false);
        ibA[base + i]     = srcA[i];
        ibB[base + i]     = srcB[i];
        ibIsP2P[base + i] = srcIsP2P[i];
    }
    __syncwarp();

    if (lane == 0)
    {
        *ibCount = base + numItems;
        releaseFlag(ibFlag);
    }
    return true;
}

/*! @brief Try to pop up to warpSize items from the interaction buffer under flag protection.
 *  @param ibA, ibB, ibIsP2P  Shared-memory IB arrays
 *  @param ibCount            Shared-memory item count
 *  @param ibFlag             Shared-memory lock flag
 *  @param outA, outB, outIsP2P  Per-lane output (valid when lane < returned popCount)
 *  @param[out] popCount      Number of items actually popped
 *  @return true if the flag was acquired and pop attempted (popCount may be 0 if IB was empty) */
__device__ __forceinline__ bool ibStackPop(
    TreeNodeIndex* __restrict__ ibA,
    TreeNodeIndex* __restrict__ ibB,
    int*           __restrict__ ibIsP2P,
    unsigned*      __restrict__ ibCount,
    unsigned*      __restrict__ ibFlag,
    TreeNodeIndex& outA,
    TreeNodeIndex& outB,
    int&           outIsP2P,
    unsigned&      popCount)
{
    auto tile = cg::coalesced_threads();
    unsigned lane = tile.thread_rank();

    popCount = 0;

    int success;
    if (lane == 0) success = acquireFlag(ibFlag);
    success = tile.shfl(success, 0);

    if (!success) return false;

    unsigned count;
    if (lane == 0) count = *ibCount;
    count = tile.shfl(count, 0u);

    unsigned take = min((unsigned)GpuConfig::warpSize, count);
    if (lane < take)
    {
        unsigned idx = count - 1 - lane;
        outA     = ibA[idx];
        outB     = ibB[idx];
        outIsP2P = ibIsP2P[idx];
    }

    if (lane == 0)
    {
        *ibCount = count - take;
        releaseFlag(ibFlag);
    }

    popCount = take;
    return true;
}


/*! @brief Compile-time thresholds for producer push / consumer pop decisions.
 *
 *  All thresholds are derived from @p StackCap_ by default so that a single
 *  number controls the entire sizing.  Override individual parameters only
 *  when fine-tuning.
 */
template<unsigned StackCap_,
         unsigned ChunkSize_       = StackCap_ / 8,
         unsigned ForcePush_       = StackCap_ * 5 / 8,
         unsigned AttemptPush_     = StackCap_ * 5 / 16,
         unsigned AttemptPop_      = StackCap_ * 3 / 16,
         unsigned ForcePop_        = StackCap_ * 1 / 16,
         unsigned TravChunkSize_   = StackCap_ / 4,
         unsigned TravForcePush_   = StackCap_ - 8 * GpuConfig::warpSize,
         unsigned TravAttemptPush_ = StackCap_ * 1 / 4,
         unsigned TravAttemptPop_  = StackCap_ * 1 / 16>
struct TraversalConfig
{
    //! @brief Shared-memory buffer capacity (node-pair slots)
    static constexpr unsigned stackCap = StackCap_;

    //! @name Interaction buffer spill parameters
    //! @{
    static constexpr unsigned chunkSize     = ChunkSize_;       //!< items per global push/pop
    static constexpr unsigned forcePush     = ForcePush_;       //!< local count above this -> must push
    static constexpr unsigned attemptPush   = AttemptPush_;     //!< local count above this -> try push
    static constexpr unsigned attemptPop    = AttemptPop_;      //!< local count below this -> try global pop
    static constexpr unsigned forcePop      = ForcePop_;        //!< local count above this -> only pop local
    //! @}

    //! @name Traversal stack spill parameters (separate from interaction spills)
    //! @{
    static constexpr unsigned travChunkSize   = TravChunkSize_;   //!< items per traversal push/pop
    static constexpr unsigned travForcePush   = TravForcePush_;   //!< stack depth above this -> must push
    static constexpr unsigned travAttemptPush = TravAttemptPush_; //!< stack depth above this -> try push
    static constexpr unsigned travAttemptPop  = TravAttemptPop_;  //!< stack depth below this -> try global pop
    //! @}
};

/*! @brief Lock-free ring queue in device memory for cross-block work sharing (interactions) */
struct GlobalWorkQueue
{
    TreeNodeIndex* nodeA;       //!< [capacity] interaction node A
    TreeNodeIndex* nodeB;       //!< [capacity] interaction node B
    int*           isP2P;       //!< [capacity] 1=p2p, 0=m2l (int for alignment)
    unsigned*      writeHead;   //!< atomic monotonic segment write counter
    unsigned*      readHead;    //!< atomic monotonic segment read counter
    unsigned*      segReady;    //!< [numSegments] per-segment ready flags
    unsigned       numSegments; //!< capacity / chunkSize
};

/*! @brief Lock-free ring queue in device memory for cross-block traversal work sharing */
struct GlobalTraversalQueue
{
    TreeNodeIndex* nodeA;       //!< [capacity] traversal node A
    TreeNodeIndex* nodeB;       //!< [capacity] traversal node B
    unsigned*      writeHead;   //!< atomic monotonic segment write counter
    unsigned*      readHead;    //!< atomic monotonic segment read counter
    unsigned*      segReady;    //!< [numSegments] per-segment ready flags
    unsigned       numSegments; //!< capacity / travChunkSize
};

__device__ __forceinline__ bool isLeaf(const TreeNodeIndex* __restrict__ childOffsets,
                                       TreeNodeIndex n)
{
    return childOffsets[n] == 0;
}

template<class MAC>
__device__ __forceinline__ bool splitSafe(const TreeNodeIndex* __restrict__ childOffsets,
                                         TreeNodeIndex a, TreeNodeIndex b,
                                         MAC&& continuation)
{
    return (!isLeaf(childOffsets, a)) &&
           (!isLeaf(childOffsets, b)) &&
           continuation(a, b);
}

__device__ __forceinline__ unsigned ceil_log8(unsigned n)
{
    if (n <= 1) return 0;
    unsigned x   = n - 1;
    unsigned msb = 31u - __clz(x);
    return (msb + 3u) / 3u;
}

__device__ __forceinline__ unsigned pow8(unsigned L)
{
    return 1u << (3u * L);
}

__device__ __forceinline__ void decode_base8_digits(unsigned rid, unsigned L, unsigned* digits)
{
    // digits[0] MS digit, digits[L-1] LS digit
    for (unsigned i = 0; i < L; ++i) {
        digits[L - 1 - i] = rid & 7u;
        rid >>= 3u;
    }
}

template<class MAC>
__device__ __forceinline__
bool assignPairBySplitting_regress(const TreeNodeIndex* __restrict__ childOffsets,
                                  TreeNodeIndex& a, TreeNodeIndex& b,
                                  unsigned rid, unsigned N,
                                  unsigned& active_count,
                                  MAC&& continuation)
{
    if (N <= 1) { active_count = 1; return rid == 0; }

    const unsigned L_target = ceil_log8(N);
    unsigned digits[16];
    if (L_target > 16) { active_count = 1; return rid == 0; }
    decode_base8_digits(rid, L_target, digits);

    // Track last safe snapshot
    TreeNodeIndex a_safe = a, b_safe = b;
    unsigned L_safe = 0;               // number of split levels safely applied
    // unsigned digits_safe[16];          // prefix digits that were safely applied

    // Try to split up to L_target levels, but stop at first unsafe.
    for (unsigned level = 0; level < L_target; ++level)
    {
        if (!splitSafe(childOffsets, a, b, continuation))
        {
            // Regress to last safe state
            a = a_safe; b = b_safe;
            break;
        }

        const unsigned oct = digits[level];

        if (a < b) a = childOffsets[a] + (TreeNodeIndex)oct;
        else       b = childOffsets[b] + (TreeNodeIndex)oct;

        a_safe = a;
        b_safe = b;
        L_safe = level + 1;
    }

    const unsigned fanout = pow8(L_safe);
    active_count = (fanout < N) ? fanout : N;

    // If we couldn't safely split even once, fanout=1 -> only rid==0 active.
    return rid < active_count;
}


/*! @brief Try to push chunkSize items from local interaction buffer to global queue.
 *  @return true if push succeeded
 *  Acquires ibFlag to safely read and shrink the IB. */
template<class TravConfig>
__device__ __forceinline__ bool tryPushToGlobal(
    GlobalWorkQueue gq,
    TreeNodeIndex* __restrict__ bufA,
    TreeNodeIndex* __restrict__ bufB,
    int*           __restrict__ bufIsP2P,
    unsigned*      __restrict__ bufCount,
    bool                        force)
{
    constexpr unsigned chunk = TravConfig::chunkSize;
    // constexpr unsigned cap   = TravConfig::stackCap;
    
    auto tile = cg::coalesced_threads();
    const unsigned lane = tile.thread_rank();
    
    unsigned wHead = 0, rHead = 0, old = 7u, seg = 0;

    do {
        if (lane == 0)
        {
            wHead = *gq.writeHead;
            rHead = *gq.readHead;
        }
        wHead = tile.shfl(wHead, 0);
        rHead = tile.shfl(rHead, 0);

        if (wHead - rHead >= gq.numSegments) return false; // full

        seg = wHead % gq.numSegments;
        if (lane == 0) {
            old = atomicCAS(&gq.segReady[seg], 0u, 1u);
            if (old == 0u) atomicAdd(gq.writeHead, 1u);
        }
        old = tile.shfl(old, 0);
    } while (force && old != 0u);

    if (old != 0u) return false; // contention

    // printf("Pushing to global: wHead=%u rHead=%u seg=%u\n", wHead, rHead, seg);

    unsigned base     = seg * chunk;
    unsigned count    = *bufCount;
    unsigned numPushed = min(chunk, count);
    unsigned srcStart  = count - numPushed; // push from the end so remainder stays at front

    for (unsigned i = lane; i < chunk; i += tile.num_threads()) {
        if (i < numPushed) {
            gq.nodeA[base + i] = bufA[srcStart + i];
            gq.nodeB[base + i] = bufB[srcStart + i];
            gq.isP2P[base + i] = bufIsP2P[srcStart + i];
        } else {
            gq.nodeA[base + i] = 0;
            gq.nodeB[base + i] = 0;
            gq.isP2P[base + i] = 2;
        }
    }
    __threadfence(); // ensure data visible to other blocks before signaling ready
    if (lane == 0) {
        atomicExch(&gq.segReady[seg], 2u);
        atomicSub(bufCount, numPushed);
    }
    __threadfence_block();
    return true;
}

/*! @brief Try to pop chunkSize items from global queue into local interaction buffer.
 *  @return true if pop succeeded
 *  Acquires ibFlag to safely grow the IB. */
template<class TravConfig>
__device__ __forceinline__ bool tryPopFromGlobal(
    GlobalWorkQueue gq,
    TreeNodeIndex* __restrict__ bufA,
    TreeNodeIndex* __restrict__ bufB,
    int*           __restrict__ bufIsP2P,
    unsigned*      __restrict__ bufCount,
    bool                        force,
    unsigned*      __restrict__ ibFlag)
{
    constexpr unsigned chunk = TravConfig::chunkSize;
    // constexpr unsigned cap   = TravConfig::stackCap;

    auto tile = cg::coalesced_threads();
    const unsigned lane = tile.thread_rank();

    unsigned wHead = 0, rHead = 0, old = 7u, seg = 0;
    do {
        if (lane == 0)
        {
            wHead = *gq.writeHead;
            rHead = *gq.readHead;
        }
        wHead = tile.shfl(wHead, 0);
        rHead = tile.shfl(rHead, 0);

        if (wHead <= rHead) return false; // empty

        seg = rHead % gq.numSegments;
        if (lane == 0) {
            old = atomicCAS(&gq.segReady[seg], 2u, 3u);
            if (old == 2u) atomicAdd(gq.readHead, 1u);
        }
        old = tile.shfl(old, 0);
    } while (force && old != 2u);

    if (old != 2u) return false; // contention (should only happen if we dont force)

    unsigned base = seg * chunk;

    // acquire flag from local buffer
    bool success = false;
    // unsigned iter = 0;
    while (!success) {
        if (lane == 0) success = acquireFlag(ibFlag);
        success = tile.shfl(success, 0);
        // if (lane == 0 && !success && (++iter % 256) == 0) printf("culprit!\n");
    }

    unsigned localCount = *bufCount;
    unsigned myValidCount = 0;
    for (unsigned j = lane; j < chunk; j += tile.num_threads())
    {
        if (gq.isP2P[base + j] != 2)
        {
            bufA[localCount + j]     = gq.nodeA[base + j];
            bufB[localCount + j]     = gq.nodeB[base + j];
            bufIsP2P[localCount + j] = gq.isP2P[base + j];
            myValidCount++;
        }
    }
    unsigned validCount = cg::reduce(tile, myValidCount, cg::plus<unsigned>());

    if (lane == 0) {
        atomicAdd(bufCount, validCount);
        releaseFlag(ibFlag);
        atomicExch(&gq.segReady[seg], 0u);
    }
    return true;
}

/*! @brief Try to push travChunkSize items from local traversal stack to global traversal queue.
 *  @return true if push succeeded
 *  Executed by lane 0 of producer warp only. */
template<class TravConfig>
__device__ __forceinline__ bool tryPushTraversalToGlobal(
    GlobalTraversalQueue gq,
    TreeNodeIndex* __restrict__ bufA,
    TreeNodeIndex* __restrict__ bufB,
    unsigned*      __restrict__ bufCount,
    bool                        force)
{
    constexpr unsigned chunk = TravConfig::travChunkSize;
    // constexpr unsigned cap   = TravConfig::stackCap;

    auto tile = cg::coalesced_threads();
    const unsigned lane = tile.thread_rank();


    unsigned wHead = 0, rHead = 0, old = 7u, seg = 0;
    do {
        if (lane == 0) {
            wHead = *gq.writeHead;
            rHead = *gq.readHead;
        }
        wHead = tile.shfl(wHead, 0);
        rHead = tile.shfl(rHead, 0);

        if (wHead - rHead >= gq.numSegments) return false; // full

        seg = wHead % gq.numSegments;
        if (lane == 0) {
            old = atomicCAS(&gq.segReady[seg], 0u, 1u);
            if (old == 0u) atomicAdd(gq.writeHead, 1u);
        }
        old = tile.shfl(old, 0);
    } while (force && old != 0u);

    if (old != 0u) return false; // contention

    unsigned base  = seg * chunk;
    unsigned count = *bufCount;

    // assert(count >= chunk && "tryPushTraversalToGlobal: bufCount < travChunkSize");
    for (unsigned i = lane; i < chunk && i < count; i += tile.num_threads())
    {
        unsigned srcIdx = count - chunk + i;
        // assert(srcIdx < cap && "tryPushTraversalToGlobal: source index OOB");
        gq.nodeA[base + i] = bufA[srcIdx];
        gq.nodeB[base + i] = bufB[srcIdx];
    }
    __threadfence(); // ensure data visible to other blocks before signaling ready
    if (lane == 0) {
        atomicSub(bufCount, chunk);
        __threadfence();
        atomicExch(&gq.segReady[seg], 2u);
    }
    return true;
}

/*! @brief Try to pop travChunkSize items from global traversal queue into local stack.
 *  @return true if pop succeeded
 *  Executed by lane 0 of producer warp only. */
template<class TravConfig>
__device__ __forceinline__ bool tryPopTraversalFromGlobal(
    GlobalTraversalQueue gq,
    TreeNodeIndex* __restrict__ bufA,
    TreeNodeIndex* __restrict__ bufB,
    unsigned*      __restrict__ bufCount,
    bool                        force)
{
    constexpr unsigned chunk = TravConfig::travChunkSize;
    // constexpr unsigned cap   = TravConfig::stackCap;


    auto tile = cg::coalesced_threads();
    const unsigned lane = tile.thread_rank();

    unsigned wHead = 0, rHead = 0, old = 7u, seg = 0;
    do{
        if (lane == 0) {
            wHead = *gq.writeHead;
            rHead = *gq.readHead;
        }
        wHead = tile.shfl(wHead, 0);
        rHead = tile.shfl(rHead, 0);

        if (wHead <= rHead) return false; // empty

        seg = rHead % gq.numSegments;
        if (lane == 0) {
            old = atomicCAS(&gq.segReady[seg], 2u, 3u);
            if (old == 2u) atomicAdd(gq.readHead, 1u);
        }
        old = tile.shfl(old, 0);
    } while (force && old != 2u);

    if (old != 2u) return false; // contention or not ready

    unsigned base = seg * chunk;
    unsigned count = *bufCount;
    // assert(count + chunk <= cap && "tryPopTraversalFromGlobal: stack overflow");
    for (unsigned i = lane; i < chunk; i += tile.num_threads())
    {
        bufA[count + i] = gq.nodeA[base + i];
        bufB[count + i] = gq.nodeB[base + i];
    }
    __threadfence_block(); // ensure shared-mem writes visible before publishing count
    if(lane == 0) {
        atomicAdd(bufCount, chunk);
        __threadfence();
        atomicExch(&gq.segReady[seg], 0u);
    }
    return true;
}

template<int numWarps, class TravConfig,
         class MAC, class M2L, class P2P>
__device__ void dualTraversalBlock(
    const TreeNodeIndex* __restrict__ childOffsets,
    TreeNodeIndex a, TreeNodeIndex b, bool producer,
    GlobalWorkQueue globalQueue,
    GlobalTraversalQueue globalTraversalQueue,
    unsigned* __restrict__ numActiveProducers,
    MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    constexpr unsigned stackCap = TravConfig::stackCap;

    cg::thread_block block = cg::this_thread_block();
    unsigned tid    = block.thread_rank();
    unsigned lane = lane_id();

    __shared__ TreeNodeIndex interactionBufferA[stackCap];
    __shared__ TreeNodeIndex interactionBufferB[stackCap];
    __shared__ int interactionBufferIsP2P[stackCap];
    __shared__ unsigned interactionBufferCount;
    __shared__ unsigned ibFlag; // 0=free, 1=held

    __shared__ unsigned globalPopFlag;

    if (tid == 0)
    {
        interactionBufferCount = 0;
        ibFlag = 0;
        globalPopFlag = 0;
    }
    block.sync();

    producer = producer && (tid < GpuConfig::warpSize);

    // ── Handle trivial cases before entering producer loop ──
    if (producer)
    {
        bool trivial = false;
        if (isLeaf(childOffsets, a) && isLeaf(childOffsets, b))
        {
            // Both leaves: p2p if criterion passes, otherwise skip.
            if (tid == 0 && continuation(a, b)) { p2p(a, b); }
            trivial = true;
        }
        else if ((isLeaf(childOffsets, a) || isLeaf(childOffsets, b)) && !continuation(a, b))
        {
            // One leaf, criterion fails: m2l and done.
            if (tid == 0) { m2l(a, b); }
            trivial = true;
        }
        // NOTE: when one is a leaf and continuation IS true, we must NOT
        // treat it as trivial — the non-leaf side needs to be split further
        // by the producer loop.

        if (trivial)
        {
            producer = false;
            __threadfence();
            if (tid == 0)
            {
                unsigned pre = atomicSub(numActiveProducers, 1u);
                // printf("[DECR blk=%u] TRIVIAL path, pre=%u\n", blockIdx.x, pre);
            }
        }
    }

    // ── Producer loop (warp 0 only, skip if trivial) ──
    if (producer)
    {
        __shared__ TreeNodeIndex localStackA[stackCap];
        __shared__ TreeNodeIndex localStackB[stackCap];
        __shared__ unsigned localStackTop;

        // Temporary interaction buffer — one outer-loop iteration produces
        // at most warpSize × 8 = 256 interaction items.  Writing into a
        // dedicated buffer first lets us verify that the main IB has room
        // *before* the merge copy, avoiding any OOB writes.
        constexpr unsigned tmpCap = GpuConfig::warpSize * 8;
        __shared__ TreeNodeIndex tmpIBA[tmpCap];
        __shared__ TreeNodeIndex tmpIBB[tmpCap];
        __shared__ int           tmpIBP2P[tmpCap];
        __shared__ unsigned      tmpIBCount;

        if (tid == 0)
        {
            localStackA[0] = a;
            localStackB[0] = b;
            localStackTop = 1;
            tmpIBCount = 0;
        }
        __syncwarp();

        // if (tid == 0) printf("[PRODUCER blk=%u] entering main loop a=%d b=%d\n", blockIdx.x, a, b);

        // unsigned __producerIterCount = 0;
        for (;;)
        {
            // ── Pop items from traversal stack, capped by available headroom ──
            // Each popped item generates up to 8 children, so we need
            // 8 * popCount free slots on the stack after the pop.
            
            unsigned room = (stackCap > localStackTop) ? (stackCap - localStackTop) : 0;
            unsigned maxPop = room / 8;
            if (maxPop == 0 && localStackTop > 0) maxPop = 1; // guarantee progress
            unsigned popCount = min(min(localStackTop, (unsigned)GpuConfig::warpSize), maxPop);
            if (tid < popCount)
            {
                unsigned readIdx = localStackTop - 1 - tid;
                // assert(readIdx < stackCap && "Producer stack pop: readIdx OOB");
                a = localStackA[readIdx];
                b = localStackB[readIdx];
            }
            if (tid == 0) localStackTop -= popCount;
            __syncwarp();

            if (tid < popCount)
            {
                TreeNodeIndex aCO = childOffsets[a];
                TreeNodeIndex bCO = childOffsets[b];
                bool aIsLeaf = (aCO == 0);
                bool bIsLeaf = (bCO == 0);
                bool splitA = (a < b && !aIsLeaf) || bIsLeaf;

                int newChildren = splitA ? 8 * (!aIsLeaf) : 8 * (!bIsLeaf);

                // NOTE: do NOT use #pragma unroll here — localStackTop and
                // interactionBufferCount are updated via atomicAdd at the end
                // of each iteration, and subsequent iterations must see the
                // fresh values.  Unrolling lets the compiler hoist/CSE the
                // plain shared-memory loads across iterations, causing
                // multiple iterations to compute the SAME write positions and
                // silently overwrite each other's data.
                for (int i = 0; i < newChildren; ++i)
                {
                    auto tile = cg::coalesced_threads();
                    TreeNodeIndex childA = splitA ? aCO + i : a;
                    TreeNodeIndex childB = splitA ? b : bCO + i;
                    bool cont = continuation(childA, childB);

                    bool childAIsLeaf = (childOffsets[childA] == 0);
                    bool childBIsLeaf = (childOffsets[childB] == 0);

                    int isTraversal = cont && !(childAIsLeaf && childBIsLeaf) ? 1 : 0;
                    int pos = cg::exclusive_scan(tile, isTraversal);
                    int total = pos + isTraversal;
                    total = tile.shfl(total, tile.num_threads() - 1);

                    // Atomic reads to get fresh values from shared memory,
                    // preventing the compiler from reusing stale registers
                    // across loop iterations.
                    unsigned curStackTop = atomicAdd(&localStackTop, 0u);
                    unsigned curTmpCount = atomicAdd(&tmpIBCount, 0u);

                    if (isTraversal)
                    {
                        unsigned writePos = pos + curStackTop;
                        // assert(writePos < stackCap && "Producer stack push: writePos OOB");
                        localStackA[writePos] = childA;
                        localStackB[writePos] = childB;
                    }
                    else
                    {
                        // interaction item -> temporary buffer (merged after inner loop)
                        pos = tile.thread_rank() - pos;
                        unsigned writeIdx = pos + curTmpCount;
                        // if (lane == 0 && writeIdx >= tmpCap) printf("Write Index: %u, Temp Cap: %u\n",writeIdx,tmpCap);
                        // assert(writeIdx < tmpCap && "Producer tmp IB: index OOB");
                        tmpIBA[writeIdx]   = childA;
                        tmpIBB[writeIdx]   = childB;
                        tmpIBP2P[writeIdx] = cont ? 1 : 0;
                    }

                    if (tile.thread_rank() == tile.num_threads() - 1)
                    {
                        // unsigned newStackTop = curStackTop + (unsigned)total;
                        unsigned newTmpCount = curTmpCount + (unsigned)(tile.num_threads() - total);
                        // assert(newStackTop <= stackCap && "Producer: localStackTop would exceed stackCap");
                        // assert(newTmpCount <= tmpCap && "Producer: tmpIBCount would exceed tmpCap");
                        atomicAdd(&localStackTop, (unsigned)total);
                        atomicAdd(&tmpIBCount, (unsigned)(tile.num_threads() - total));
                    }
                }
            }
            __syncwarp(); // ensure all inner-loop writes to tmpIB/localStack are visible

            // if (tid == 0 && (++__producerIterCount % 4096) == 0)
            //     printf("[PRODUCER blk=%u] main iter=%u lstk=%u tmpIB=%u ibCount=%u producers=%u\n",
            //            blockIdx.x, __producerIterCount, localStackTop,
            //            atomicAdd(&tmpIBCount, 0u), atomicAdd(&interactionBufferCount, 0u),
            //            atomicAdd(numActiveProducers, 0u));

            // unsigned __drainIter = 0;
            while (tmpIBCount > 0) {

                // auto tile = cg::coalesced_threads();
                // if (tile.num_threads() != GpuConfig::warpSize) {
                //     if (tile.thread_rank() == 0) assert(false);
                // }

                // Force reload from shared memory — interactionBufferCount is
                // modified by consumer warps; without an atomic read the
                // compiler may cache a stale value in a register, causing the
                // producer to spin forever on the wrong branch.
                unsigned ibCount;
                if (lane == 0) ibCount = interactionBufferCount;
                ibCount = __shfl_sync(0xFFFFFFFFu, ibCount, 0);

                if (ibCount > TravConfig::forcePush || (ibCount + tmpIBCount > TravConfig::stackCap)) {
                    // if (lane == 0) printf("Block attempting force push\n");
                    if (tryPushToGlobal<TravConfig>(globalQueue, tmpIBA, tmpIBB, tmpIBP2P, &tmpIBCount, true)) {}
                    else if (localStackTop > TravConfig::travAttemptPush) {
                        tryPushTraversalToGlobal<TravConfig>(globalTraversalQueue, localStackA, localStackB, &localStackTop, false);
                    }
                } else if (ibCount > TravConfig::attemptPush) {
                    // if (lane == 0) printf("Block attempting push\n");
                    if (tryPushToGlobal<TravConfig>(globalQueue, tmpIBA, tmpIBB, tmpIBP2P, &tmpIBCount, false)) {}
                    else {
                        if (ibStackPush<TravConfig::stackCap>(interactionBufferA, interactionBufferB, interactionBufferIsP2P,
                                                              &interactionBufferCount, &ibFlag,
                                                              tmpIBA, tmpIBB, tmpIBP2P, tmpIBCount)) {
                            if (lane == 0) atomicExch(&tmpIBCount, 0u);
                        }
                    }
                    // if (lane == 0) printf("Block attempted push\n");
                } else {
                    // if (lane == 0) printf("Block attempting push to local Stack \n");
                    if (ibStackPush<TravConfig::stackCap>(interactionBufferA, interactionBufferB, interactionBufferIsP2P,
                                                          &interactionBufferCount, &ibFlag,
                                                          tmpIBA, tmpIBB, tmpIBP2P, tmpIBCount)) {
                        // if (lane == 0) printf("Successful\n");
                        if (lane == 0) atomicExch(&tmpIBCount, 0u);
                    } 
                    else if (localStackTop > TravConfig::travForcePush) {
                        tryPushTraversalToGlobal<TravConfig>(globalTraversalQueue, localStackA, localStackB, &localStackTop, false);
                    }
                    // if (lane == 0) printf("Block completed push to local Stack \n");
                }
                // __threadfence_block();
              

                // if (lane == 0 && (++__drainIter % 4096) == 0)
                //     printf("[DRAIN blk=%u] iter=%u tmpIB=%u ibCount=%u lstk=%u gqW=%u gqR=%u producers=%u\n",
                //            blockIdx.x, __drainIter,
                //            atomicAdd(&tmpIBCount, 0u), atomicAdd(&interactionBufferCount, 0u),
                //            atomicAdd(&localStackTop, 0u),
                //            atomicAdd(globalQueue.writeHead, 0u), atomicAdd(globalQueue.readHead, 0u),
                //            atomicAdd(numActiveProducers, 0u));
            }
            __syncwarp();

            // traversal push and pop logic
            if (localStackTop >= TravConfig::travForcePush || (localStackTop + GpuConfig::warpSize * 8 >= stackCap)) {
                tryPushTraversalToGlobal<TravConfig>(globalTraversalQueue, localStackA, localStackB, &localStackTop, true);
            } else if (localStackTop > TravConfig::travAttemptPush) {
                tryPushTraversalToGlobal<TravConfig>(globalTraversalQueue, localStackA, localStackB, &localStackTop, false);
            } else if (localStackTop == 0) {
                // if (lane == 0) printf("Enter\n");
                if (tryPopTraversalFromGlobal<TravConfig>(globalTraversalQueue, localStackA, localStackB, &localStackTop, true)) {}
                else {
                    // if (lane == 0) printf("Exit\n");
                    break;
                }
            } else if (localStackTop <= TravConfig::travAttemptPop) {
                tryPopTraversalFromGlobal<TravConfig>(globalTraversalQueue, localStackA, localStackB, &localStackTop, false);
            }
            __syncwarp();
        }
        // if (tid == 0) printf("[PRODUCER blk=%u] exiting main loop\n", blockIdx.x);
        // __threadfence(); // ensure all IB/global-queue writes visible before signaling done
        if(lane == 0)
        {
            unsigned pre = atomicSub(numActiveProducers, 1u);
            // printf("[DECR blk=%u] END-OF-LOOP, pre=%u\n", blockIdx.x, pre);
        }
    }

    // unsigned consumerSpinCount = 0;
    auto tile = cg::coalesced_threads();

    auto wRef = cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*globalQueue.writeHead);
    auto rRef = cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*globalQueue.readHead);
    auto pRef = cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*numActiveProducers);
    for (;;)
    {
        // consumerSpinCount++;
        // if (lane == 0 && (consumerSpinCount & 0x3FFFF) == 0)
        //     printf("[Consumer blk=%u warp=%u] spin %u: ibCount=%u ibFlag=%u wHead=%u rHead=%u producers=%u\n",
        //            blockIdx.x, tid / GpuConfig::warpSize, consumerSpinCount,
        //            atomicAdd(&interactionBufferCount, 0u), atomicAdd(&ibFlag, 0u),
        //            atomicAdd(globalQueue.writeHead, 0u),
        //            atomicAdd(globalQueue.readHead, 0u),
        //            atomicAdd(numActiveProducers, 0u));

        // ── Try to pop items from local IB (flag-guarded) ──
        TreeNodeIndex itemA, itemB;
        int itemIsP2P;
        unsigned popCount = 0;

        unsigned numLocal;
        if (lane == 0) {
            numLocal = interactionBufferCount;
        }
        numLocal = tile.shfl(numLocal, 0);

        if (numLocal < TravConfig::attemptPop) {
            bool success = false;
            if (lane == 0) success = acquireFlag(&globalPopFlag);
            success = tile.shfl(success, 0);

            if(success) {
                bool force = numLocal <= TravConfig::forcePop;
                tryPopFromGlobal<TravConfig>(globalQueue, interactionBufferA, interactionBufferB, interactionBufferIsP2P, &interactionBufferCount, force, &ibFlag);
                if (lane == 0) releaseFlag(&globalPopFlag);
            }
        }

        // __syncwarp();

        if (lane == 0) {
            numLocal = interactionBufferCount;
        }
        numLocal = tile.shfl(numLocal, 0);

        if (numLocal > 0) {
            ibStackPop(interactionBufferA, interactionBufferB, interactionBufferIsP2P, &interactionBufferCount, &ibFlag, itemA, itemB, itemIsP2P, popCount);
        }

        // __syncwarp();

        // ── Process popped items ──
        if (lane < popCount)
        {
            // if (!(itemIsP2P == 0 || itemIsP2P == 1)) {printf("Item: %d\n", itemIsP2P);}
            // assert((itemIsP2P == 0 || itemIsP2P == 1) && "Consumer: itemIsP2P corrupted");
            if (itemIsP2P)  p2p(itemA, itemB);
            else            m2l(itemA, itemB);
        }
        // __syncwarp();

        unsigned localCount, wHead, rHead, producers;

        bool terminate = false;
        if (popCount == 0) {
            if (lane == 0) {
                localCount = interactionBufferCount;  // smem: plain load if synchronized
                producers = pRef.load(cuda::memory_order_acquire);
                wHead     = wRef.load(cuda::memory_order_acquire);
                rHead     = rRef.load(cuda::memory_order_acquire);
                terminate = (localCount == 0 && wHead <= rHead && producers == 0);
            }
            terminate = tile.shfl(terminate, 0);
        }
        if (terminate) break;
    }

}

template<int numWarps, class TravConfig,
         class MAC, class M2L, class P2P>
__device__ void dualTraversalTBC(const TreeNodeIndex* __restrict__ childOffsets,
                                 TreeNodeIndex a, TreeNodeIndex b,
                                 GlobalWorkQueue globalQueue,
                                 GlobalTraversalQueue globalTraversalQueue,
                                 unsigned* __restrict__ numActiveProducers,
                                 MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    const unsigned block_in_cluster = block_rank_in_cluster();
    const unsigned blocksPerCluster = blocks_per_cluster_runtime();

    unsigned active_blocks = 0;
    bool producer =
        assignPairBySplitting_regress(childOffsets, a, b,
                                      block_in_cluster, blocksPerCluster,
                                      active_blocks,
                                      std::forward<MAC>(continuation));

    // non-producer blocks decrement active count immediately
    if (!producer && cg::this_thread_block().thread_rank() == 0)
    {
        unsigned pre = atomicSub(numActiveProducers, 1u);
        // printf("[DECR blk=%u] TBC non-producer, pre=%u\n", blockIdx.x, pre);
    }

    dualTraversalBlock<numWarps, TravConfig>(
        childOffsets, a, b, producer,
        globalQueue, globalTraversalQueue, numActiveProducers,
        std::forward<MAC>(continuation),
        std::forward<M2L>(m2l),
        std::forward<P2P>(p2p));
}

template<int numWarps, class TravConfig,
         class MAC, class M2L, class P2P>
__device__ void dualTraversalGPU(const TreeNodeIndex* __restrict__ childOffsets,
                                 TreeNodeIndex rootA, TreeNodeIndex rootB,
                                 GlobalWorkQueue globalQueue,
                                 GlobalTraversalQueue globalTraversalQueue,
                                 unsigned* __restrict__ numActiveProducers,
                                 MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    const unsigned cluster_id  = cluster_rank_in_grid();
    const unsigned numClusters = num_clusters_runtime();

    TreeNodeIndex a = rootA, b = rootB;

    unsigned active_clusters = 0;
    const bool active_cluster =
        assignPairBySplitting_regress(childOffsets, a, b,
                                      cluster_id, numClusters,
                                      active_clusters,
                                      std::forward<MAC>(continuation));

    if (!active_cluster)
    {
        if (cg::this_thread_block().thread_rank() == 0)
        {
            unsigned pre = atomicSub(numActiveProducers, 1u);
            // printf("[DECR blk=%u] GPU non-active cluster, pre=%u\n", blockIdx.x, pre);
        }
        return;
    }

    dualTraversalTBC<numWarps, TravConfig>(
        childOffsets, a, b,
        globalQueue, globalTraversalQueue, numActiveProducers,
        std::forward<MAC>(continuation),
        std::forward<M2L>(m2l),
        std::forward<P2P>(p2p));
}

} // namespace cstone
