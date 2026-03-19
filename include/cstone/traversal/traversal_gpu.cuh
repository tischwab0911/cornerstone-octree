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
#include <cuda/ptx>
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

constexpr unsigned kProducerWarpsPerBlock = 2;

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

    return totalBlocks;
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

/*! @brief Bounded backoff for hot spin loops to reduce contention pressure. */
__device__ __forceinline__ void spinBackoff(unsigned iter)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    if (iter >= 8)
    {
        unsigned shift = iter - 8;
        if (shift > 8) shift = 8;
        __nanosleep(1u << shift);
    }
#else
    (void)iter;
#endif
}


template<unsigned cap>
__device__ __forceinline__ bool ibQueuePush(
    TreeNodeIndex* __restrict__ ibA,
    TreeNodeIndex* __restrict__ ibB,
    int*           __restrict__ ibIsP2P,
    unsigned*      __restrict__ ibHead,
    unsigned*      __restrict__ ibTail,
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

    unsigned count = 0u, tail = 0u;
    if (lane == 0)
    {
        count = *ibCount;
        tail  = *ibTail;
    }
    count = tile.shfl(count, 0);
    tail  = tile.shfl(tail, 0);

    if (count + numItems > cap)
    {
        if (lane == 0) releaseFlag(ibFlag);
        return false;
    }

    unsigned first = min(numItems, cap - tail);
    for (unsigned i = lane; i < first; i += GpuConfig::warpSize)
    {
        unsigned dst = tail + i;
        ibA[dst]     = srcA[i];
        ibB[dst]     = srcB[i];
        ibIsP2P[dst] = srcIsP2P[i];
    }

    unsigned second = numItems - first;
    for (unsigned i = lane; i < second; i += GpuConfig::warpSize)
    {
        ibA[i]     = srcA[first + i];
        ibB[i]     = srcB[first + i];
        ibIsP2P[i] = srcIsP2P[first + i];
    }
    __syncwarp();

    if (lane == 0)
    {
        unsigned newTail = tail + numItems;
        if (newTail >= cap) newTail -= cap;
        *ibTail  = newTail;
        *ibCount = count + numItems;
        releaseFlag(ibFlag);
    }
    return true;
}

/*! @brief Try to dequeue up to warpSize items from the shared-memory interaction queue.
 *  @param ibA, ibB, ibIsP2P Shared-memory queue payload arrays
 *  @param ibHead, ibTail, ibCount Queue state
 *  @param ibFlag Shared-memory lock flag
 *  @param outA, outB, outIsP2P Per-lane output (valid when lane < returned popCount)
 *  @param[out] popCount Number of items actually dequeued
 *  @return true if lock was acquired and dequeue attempted */
template<unsigned cap>
__device__ __forceinline__ bool ibQueuePop(
    TreeNodeIndex* __restrict__ ibA,
    TreeNodeIndex* __restrict__ ibB,
    int*           __restrict__ ibIsP2P,
    unsigned*      __restrict__ ibHead,
    unsigned*      __restrict__ ibTail,
    unsigned*      __restrict__ ibCount,
    unsigned*      __restrict__ ibFlag,
    TreeNodeIndex& outA,
    TreeNodeIndex& outB,
    int&           outIsP2P,
    unsigned&      popCount)
{
    auto tile = cg::coalesced_threads();
    unsigned lane = tile.thread_rank();

    (void)ibTail;
    popCount = 0;

    int success;
    if (lane == 0) success = acquireFlag(ibFlag);
    success = tile.shfl(success, 0);
    if (!success) return false;

    unsigned count = 0u, head = 0u;
    if (lane == 0)
    {
        count = *ibCount;
        head  = *ibHead;
    }
    count = tile.shfl(count, 0);
    head  = tile.shfl(head, 0);

    unsigned take = min((unsigned)GpuConfig::warpSize, count);
    unsigned first = min(take, cap - head);
    if (lane < first)
    {
        unsigned idx = head + lane;
        outA     = ibA[idx];
        outB     = ibB[idx];
        outIsP2P = ibIsP2P[idx];
    }
    else if (lane < take)
    {
        unsigned idx = lane - first;
        outA     = ibA[idx];
        outB     = ibB[idx];
        outIsP2P = ibIsP2P[idx];
    }

    if (lane == 0)
    {
        unsigned newHead = head + take;
        if (newHead >= cap) newHead -= cap;
        *ibHead  = newHead;
        *ibCount = count - take;
        __threadfence_block();
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
         unsigned ChunkSize_       = 128,
         unsigned ForcePush_       = 640,
         unsigned AttemptPush_     = 512,
         unsigned AttemptPop_      = 160,
         unsigned ForcePop_        = 64,
         unsigned TravChunkSize_   = 256,
         unsigned TravForcePush_   = StackCap_ - 8 * GpuConfig::warpSize,
         unsigned TravAttemptPush_ = 256,
         unsigned TravAttemptPop_  = 32>
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
    unsigned*      segCount;    //!< [numSegments] valid item count per segment
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

__device__ __forceinline__ TreeNodeIndex childOffsetLoad(const TreeNodeIndex* __restrict__ childOffsets,
                                                         TreeNodeIndex n)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return __ldg(childOffsets + n);
#else
    return childOffsets[n];
#endif
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
bool assignPairByFrontierDeterministic(const TreeNodeIndex* __restrict__ childOffsets,
                                       TreeNodeIndex& a, TreeNodeIndex& b,
                                       unsigned rid, unsigned N,
                                       unsigned& active_count,
                                       MAC&& continuation)
{
    constexpr unsigned kMaxSeedPairs = 256;

    if (N == 0)
    {
        active_count = 0;
        return false;
    }

    cg::thread_block block = cg::this_thread_block();
    unsigned tid = block.thread_rank();

    __shared__ TreeNodeIndex frontierA0[kMaxSeedPairs];
    __shared__ TreeNodeIndex frontierB0[kMaxSeedPairs];
    __shared__ TreeNodeIndex frontierA1[kMaxSeedPairs];
    __shared__ TreeNodeIndex frontierB1[kMaxSeedPairs];

    __shared__ unsigned frontierSize;
    __shared__ unsigned useFirst;
    __shared__ unsigned continueSplit;

    __shared__ unsigned canSplit[kMaxSeedPairs];
    __shared__ unsigned splitAFlag[kMaxSeedPairs];
    __shared__ TreeNodeIndex splitAOffset[kMaxSeedPairs];
    __shared__ TreeNodeIndex splitBOffset[kMaxSeedPairs];

    unsigned target = min(N, kMaxSeedPairs);

    if (tid == 0)
    {
        frontierA0[0] = a;
        frontierB0[0] = b;
        frontierSize = 1;
        useFirst = 1;
    }
    block.sync();

    while (true)
    {
        unsigned localSize = frontierSize;
        auto readA = useFirst ? frontierA0 : frontierA1;
        auto readB = useFirst ? frontierB0 : frontierB1;

        for (unsigned i = tid; i < localSize; i += block.size())
        {
            TreeNodeIndex ai = readA[i];
            TreeNodeIndex bi = readB[i];

            TreeNodeIndex aCo = childOffsetLoad(childOffsets, ai);
            TreeNodeIndex bCo = childOffsetLoad(childOffsets, bi);
            bool aLeaf = (aCo == 0);
            bool bLeaf = (bCo == 0);
            bool cont = continuation(ai, bi);

            bool can = false;
            bool splitA = false;
            if (cont)
            {
                if (aLeaf && bLeaf)
                {
                    can = false;
                }
                else if (aLeaf)
                {
                    can = true;
                    splitA = false;
                }
                else if (bLeaf)
                {
                    can = true;
                    splitA = true;
                }
                else
                {
                    can = true;
                    splitA = (ai < bi);
                }
            }

            canSplit[i] = can ? 1u : 0u;
            splitAFlag[i] = splitA ? 1u : 0u;
            splitAOffset[i] = aCo;
            splitBOffset[i] = bCo;
        }
        block.sync();

        if (tid == 0)
        {
            unsigned splitCandidates = 0;
            for (unsigned i = 0; i < localSize; ++i) splitCandidates += canSplit[i];

            unsigned splitBudget = (target > localSize) ? (target - localSize) / 7u : 0u;
            unsigned chosenSplits = min(splitCandidates, splitBudget);

            if (chosenSplits == 0)
            {
                continueSplit = 0;
            }
            else
            {
                auto writeA = useFirst ? frontierA1 : frontierA0;
                auto writeB = useFirst ? frontierB1 : frontierB0;

                unsigned out = 0;
                unsigned used = 0;
                for (unsigned i = 0; i < localSize; ++i)
                {
                    TreeNodeIndex ai = readA[i];
                    TreeNodeIndex bi = readB[i];

                    if (canSplit[i] && used < chosenSplits)
                    {
                        bool splitA = (splitAFlag[i] != 0u);
                        TreeNodeIndex aCo = splitAOffset[i];
                        TreeNodeIndex bCo = splitBOffset[i];
                        for (unsigned oct = 0; oct < 8; ++oct)
                        {
                            writeA[out] = splitA ? (aCo + TreeNodeIndex(oct)) : ai;
                            writeB[out] = splitA ? bi : (bCo + TreeNodeIndex(oct));
                            ++out;
                        }
                        ++used;
                    }
                    else
                    {
                        writeA[out] = ai;
                        writeB[out] = bi;
                        ++out;
                    }
                }

                frontierSize = out;
                useFirst ^= 1u;
                continueSplit = 1;
            }
        }
        block.sync();

        if (!continueSplit) break;
    }

    active_count = frontierSize;

    bool active = rid < active_count;
    if (active)
    {
        auto finalA = useFirst ? frontierA0 : frontierA1;
        auto finalB = useFirst ? frontierB0 : frontierB1;
        a = finalA[rid];
        b = finalB[rid];
    }
    return active;
}


/*! @brief Try to push up to chunkSize items from a local interaction buffer to the global queue.
 *  @return true if push succeeded
 *  Vyukov-style MPMC: claim a unique slot via CAS on writeHead, then use 0/1 per-segment flag.
 *  Uses per-segment counts to avoid sentinel padding and scan-on-pop. */
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

    auto tile = cg::coalesced_threads();
    const unsigned lane = tile.thread_rank();

    // Phase 1: claim a unique write slot via CAS on writeHead
    unsigned wHead = 0;
    unsigned rHead = 0;
    bool claimed = false;
    do {
        
        if (lane == 0)
        {
            wHead = *gq.writeHead;
            rHead = *gq.readHead;
        }
        wHead = tile.shfl(wHead, 0);
        rHead = tile.shfl(rHead, 0);

        if (wHead - rHead >= gq.numSegments) return false; // full

        if (lane == 0) {
            claimed = (atomicCAS(gq.writeHead, wHead, wHead + 1) == wHead);
        }
        claimed = tile.shfl(claimed, 0);
    } while (force && !claimed);

    if (!claimed) return false;

    // Phase 2: wait for our segment to become free (bounded: only one specific pop can hold it)
    unsigned seg = wHead % gq.numSegments;
    if (lane == 0) {
        for (unsigned spin = 0;; ++spin)
        {
            if (atomicAdd(&gq.segReady[seg], 0u) == 0u) break;
            spinBackoff(spin);
        }
    }
    tile.sync();

    // Phase 3: write data
    unsigned base      = seg * chunk;

    unsigned count;
    if (lane == 0) count = *bufCount;
    count = tile.shfl(count, 0);

    unsigned numPushed = min(chunk, count);
    unsigned srcStart  = count - numPushed;

    for (unsigned i = lane; i < numPushed; i += tile.num_threads()) {
            gq.nodeA[base + i] = bufA[srcStart + i];
            gq.nodeB[base + i] = bufB[srcStart + i];
            gq.isP2P[base + i] = bufIsP2P[srcStart + i];
    }

    // Phase 4: publish data and update count
    __threadfence();
    if (lane == 0) {
        *bufCount = count - numPushed;
        gq.segCount[seg] = numPushed;
        __threadfence();
        atomicExch(&gq.segReady[seg], 1u);
    }
    return true;
}

/*! @brief Try to pop chunkSize items from global queue into local interaction buffer.
 *  @return true if pop succeeded
 *  Vyukov-style MPMC: claim a unique slot via CAS on readHead, then use 0/1 per-segment flag.
 *  Acquires ibFlag to safely grow the IB. */
template<class TravConfig>
__device__ __forceinline__ bool tryPopFromGlobal(
    GlobalWorkQueue gq,
    TreeNodeIndex* __restrict__ bufA,
    TreeNodeIndex* __restrict__ bufB,
    int*           __restrict__ bufIsP2P,
    unsigned*      __restrict__ bufHead,
    unsigned*      __restrict__ bufTail,
    unsigned*      __restrict__ bufCount,
    bool                        force,
    unsigned*      __restrict__ ibFlag)
{
    constexpr unsigned chunk = TravConfig::chunkSize;

    auto tile = cg::coalesced_threads();
    const unsigned lane = tile.thread_rank();

    // Phase 1: claim a unique read slot via CAS on readHead
    // unsigned pos;
    unsigned rHead = 0;
    unsigned wHead = 0;
    bool claimed = false;
    do {
        // unsigned rHead = 0, wHead = 0;
        if (lane == 0)
        {
            rHead = *gq.readHead;
            wHead = *gq.writeHead;
        }
        rHead = tile.shfl(rHead, 0);
        wHead = tile.shfl(wHead, 0);

        if (wHead <= rHead) return false; // empty

        if (lane == 0) {
            claimed = (atomicCAS(gq.readHead, rHead, rHead + 1) == rHead);
        }
        claimed = tile.shfl(claimed, 0);
        // pos = rHead;
    } while (force && !claimed);

    if (!claimed) return false;

    // Phase 2: wait for our segment's data to be published (bounded: one specific push)
    unsigned seg = rHead % gq.numSegments;
    if (lane == 0) {
        for (unsigned spin = 0;; ++spin)
        {
            if (atomicAdd(&gq.segReady[seg], 0u) == 1u) break;
            spinBackoff(spin);
        }
    }
    tile.sync();
    __threadfence();

    unsigned base = seg * chunk;

    // Exact payload length for this segment, published by producer.
    unsigned validCount = 0;
    if (lane == 0) validCount = gq.segCount[seg];
    validCount = tile.shfl(validCount, 0);

    // Acquire ibFlag and wait for enough local IB space before appending.
    unsigned tail = 0;
    unsigned count = 0;
    bool haveSpace = false;
    while (!haveSpace)
    {
        bool success = false;
        if (lane == 0) success = acquireFlag(ibFlag);
        success = tile.shfl(success, 0);
        if (!success)
        {
            if (lane == 0) spinBackoff(8);
            continue;
        }

        if (lane == 0)
        {
            tail = *bufTail;
            count = *bufCount;
            haveSpace = (count + validCount <= TravConfig::stackCap);
            if (!haveSpace) releaseFlag(ibFlag);
            if (!haveSpace) spinBackoff(8);
        }
        tail = tile.shfl(tail, 0);
        count = tile.shfl(count, 0);
        haveSpace = tile.shfl(haveSpace, 0);
    }

    // Phase 3: enqueue valid items into the local ring queue tail.
    for (unsigned j = lane; j < validCount; j += tile.num_threads())
    {
        unsigned writeIdx = tail + j;
        if (writeIdx >= TravConfig::stackCap) writeIdx -= TravConfig::stackCap;
        bufA[writeIdx]     = gq.nodeA[base + j];
        bufB[writeIdx]     = gq.nodeB[base + j];
        bufIsP2P[writeIdx] = gq.isP2P[base + j];
    }

    // Phase 4: release ibFlag and free the segment
    __threadfence_block();
    if (lane == 0) {
        unsigned newTail = tail + validCount;
        if (newTail >= TravConfig::stackCap) newTail -= TravConfig::stackCap;
        *bufTail  = newTail;
        *bufCount = count + validCount;
        (void)bufHead;
        releaseFlag(ibFlag);
        gq.segCount[seg] = 0u;
        __threadfence();
        atomicExch(&gq.segReady[seg], 0u);
    }
    return true;
}

/*! @brief Try to push travChunkSize items from local traversal stack to global traversal queue.
 *  @return true if push succeeded
 *  Vyukov-style MPMC: claim a unique slot via CAS on writeHead, then use 0/1 per-segment flag.
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

    auto tile = cg::coalesced_threads();
    const unsigned lane = tile.thread_rank();

    // Phase 1: claim a unique write slot via CAS on writeHead
    bool claimed = false;
    unsigned rHead = 0;
    unsigned wHead = 0;
    do {
        if (lane == 0) {
            wHead = *gq.writeHead;
            rHead = *gq.readHead;
        }
        wHead = tile.shfl(wHead, 0);
        rHead = tile.shfl(rHead, 0);

        if (wHead - rHead >= gq.numSegments) return false; // full

        if (lane == 0) {
            claimed = (atomicCAS(gq.writeHead, wHead, wHead + 1) == wHead);
        }
        claimed = tile.shfl(claimed, 0);
    } while (force && !claimed);

    if (!claimed) return false;

    // Phase 2: wait for our segment to become free (bounded: only one specific pop can hold it)
    unsigned seg = wHead % gq.numSegments;
    if (lane == 0) {
        for (unsigned spin = 0;; ++spin)
        {
            if (atomicAdd(&gq.segReady[seg], 0u) == 0u) break;
            spinBackoff(spin);
        }
    }
    tile.sync();

    // Phase 3: write data
    unsigned base  = seg * chunk;
    unsigned count = *bufCount;
    for (unsigned i = lane; i < chunk && i < count; i += tile.num_threads())
    {
        unsigned srcIdx = count - chunk + i;
        // assert(srcIdx < TravConfig::stackCap && "tryPushTraversalToGlobal: srcIdx OOB");
        gq.nodeA[base + i] = bufA[srcIdx];
        gq.nodeB[base + i] = bufB[srcIdx];
    }

    // Phase 4: publish data and update count
    __threadfence();
    if (lane == 0) {
        *bufCount = count - chunk;
        __threadfence();
        atomicExch(&gq.segReady[seg], 1u);
    }
    return true;
}

/*! @brief Try to pop travChunkSize items from global traversal queue into local stack.
 *  @return true if pop succeeded
 *  Vyukov-style MPMC: claim a unique slot via CAS on readHead, then use 0/1 per-segment flag.
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

    auto tile = cg::coalesced_threads();
    const unsigned lane = tile.thread_rank();

    // Phase 1: claim a unique read slot via CAS on readHead
    unsigned rHead = 0;
    unsigned wHead = 0;
    bool claimed = false;
    do {
        if (lane == 0) {
            rHead = *gq.readHead;
            wHead = *gq.writeHead;
        }
        rHead = tile.shfl(rHead, 0);
        wHead = tile.shfl(wHead, 0);

        if (wHead <= rHead) return false; // empty

        if (lane == 0) {
            claimed = (atomicCAS(gq.readHead, rHead, rHead + 1) == rHead);
        }
        claimed = tile.shfl(claimed, 0);
    } while (force && !claimed);

    if (!claimed) return false;

    // Phase 2: wait for our segment's data to be published (bounded: one specific push)
    unsigned seg = rHead % gq.numSegments;
    if (lane == 0) {
        for (unsigned spin = 0;; ++spin)
        {
            if (atomicAdd(&gq.segReady[seg], 0u) == 1u) break;
            spinBackoff(spin);
        }
    }
    tile.sync();
    __threadfence();

    // Phase 3: read data
    unsigned base = seg * chunk;
    unsigned count = *bufCount;
    for (unsigned i = lane; i < chunk; i += tile.num_threads())
    {
        // assert(count + i < TravConfig::stackCap && "tryPopTraversalFromGlobal: write OOB");
        bufA[count + i] = gq.nodeA[base + i];
        bufB[count + i] = gq.nodeB[base + i];
    }

    // Phase 4: update count and free the segment
    if (lane == 0) {
        *bufCount = count + chunk;
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
    constexpr unsigned producerWarpCount = kProducerWarpsPerBlock;
    static_assert(numWarps >= producerWarpCount, "dualTraversalBlock requires at least 2 warps");
    constexpr unsigned tmpCap = GpuConfig::warpSize * 8;

    cg::thread_block block = cg::this_thread_block();
    unsigned tid    = block.thread_rank();
    unsigned lane   = lane_id();
    unsigned warpId = tid / GpuConfig::warpSize;

    __shared__ TreeNodeIndex interactionBufferA[stackCap];
    __shared__ TreeNodeIndex interactionBufferB[stackCap];
    __shared__ int interactionBufferIsP2P[stackCap];
    __shared__ unsigned interactionBufferHead;
    __shared__ unsigned interactionBufferTail;
    __shared__ unsigned interactionBufferCount;
    __shared__ unsigned ibFlag; // 0=free, 1=held


    __shared__ TreeNodeIndex localStackA[producerWarpCount][stackCap];
    __shared__ TreeNodeIndex localStackB[producerWarpCount][stackCap];
    __shared__ unsigned localStackTop[producerWarpCount];

    __shared__ TreeNodeIndex tmpIBA[producerWarpCount][tmpCap];
    __shared__ TreeNodeIndex tmpIBB[producerWarpCount][tmpCap];
    __shared__ int           tmpIBP2P[producerWarpCount][tmpCap];
    __shared__ unsigned      tmpIBCount[producerWarpCount];

    __shared__ TreeNodeIndex smem_parentACO[producerWarpCount][GpuConfig::warpSize];
    __shared__ TreeNodeIndex smem_parentBCO[producerWarpCount][GpuConfig::warpSize];
    __shared__ TreeNodeIndex smem_parentA[producerWarpCount][GpuConfig::warpSize];
    __shared__ TreeNodeIndex smem_parentB[producerWarpCount][GpuConfig::warpSize];
    __shared__ unsigned      smem_parentSplitA[producerWarpCount][GpuConfig::warpSize];
    __shared__ unsigned      smem_parentConstLeaf[producerWarpCount][GpuConfig::warpSize];

    __shared__ unsigned globalPopFlag;
    __shared__ unsigned trivialShared;
    auto pRef = cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*numActiveProducers);
    

    if (tid == 0)
    {
        interactionBufferHead = 0;
        interactionBufferTail = 0;
        interactionBufferCount = 0;
        ibFlag = 0;
        globalPopFlag = 0;
        trivialShared = 0;
        for (unsigned w = 0; w < producerWarpCount; ++w)
        {
            localStackTop[w] = 0;
            tmpIBCount[w] = 0;
        }
    }
    block.sync();

    bool producerBlock = producer;
    bool isProducerWarp = producerBlock && (warpId < producerWarpCount);
    bool trivial = !producerBlock;
    producer = isProducerWarp && !trivial;

    // ── Handle trivial cases before entering producer loop ──
    if (producerBlock && warpId == 0)
    {
        if (!continuation(a, b))
        {
            // MAC passes (well-separated): M2L regardless of leaf status
            if (tid == 0) { m2l(a, b); }
            trivial = true;
        }
        else if (isLeaf(childOffsets, a) && isLeaf(childOffsets, b))
        {
            // Both leaves, MAC fails: P2P
            if (tid == 0) { p2p(a, b); }
            trivial = true;
        }
        // else: MAC fails, at least one internal → producer loop descends

        if (trivial)
        {
            // producer = false;
            __threadfence();
            if (tid == 0)
            {
                // printf("[P EXIT blk=%u] trivial (pair a=%d b=%d)\n", blockIdx.x, a, b);
                trivialShared = 1;
                unsigned pre = atomicSub(numActiveProducers, producerWarpCount);
                // printf("[DECR blk=%u] TRIVIAL path, pre=%u\n", blockIdx.x, pre);
            }
        }
    }

    block.sync();
    trivial = trivial || (trivialShared != 0);

    producer = isProducerWarp && !trivial;

    // ── Producer loop (warp 0 only, skip if trivial) ──
    if (producer)
    {

        const unsigned producerId = warpId;

        auto wRef = cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*globalTraversalQueue.writeHead);
        auto rRef = cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*globalTraversalQueue.readHead);

        bool currentlySignaledAsActive = !trivial;

        if (lane == 0 && producerId == 0 && currentlySignaledAsActive)
        {
            localStackA[producerId][0] = a;
            localStackB[producerId][0] = b;
            localStackTop[producerId] = 1;
            tmpIBCount[producerId] = 0;
        }
        else if (lane == 0)
        {
            localStackTop[producerId] = 0;
            tmpIBCount[producerId] = 0;
        }
        __syncwarp();

        // if (tid == 0) printf("[PRODUCER blk=%u] entering main loop a=%d b=%d\n", blockIdx.x, a, b);

        // constexpr unsigned __producerMaxIter = 10000000; // DEBUG: force-exit after this many iterations
        // unsigned __producerIterCount = 0;
        for (;;)
        {
            // if (__producerIterCount >= __producerMaxIter)
            // {
            //     if (lane == 0 && producerId == 0)
            //         printf("[P blk=%u] TIMEOUT after %u iters, lstk=%u tmpIB=%u ibCnt=%u prod=%u\n",
            //                blockIdx.x, __producerIterCount, localStackTop[producerId],
            //                tmpIBCount[producerId],
            //                atomicAdd(&interactionBufferCount, 0u),
            //                atomicAdd(numActiveProducers, 0u));
            //     // Signal this producer as inactive before bailing
            //     if (lane == 0 && currentlySignaledAsActive)
            //     {
            //         atomicSub(numActiveProducers, 1u);
            //         __threadfence();
            //     }
            //     break;
            // }
            // ── Pop items from traversal stack, capped by available headroom ──
            // Each popped item generates up to 8 children, so we need
            // 8 * popCount free slots on the stack after the pop.
            unsigned curStackTop_;
            if (lane == 0) curStackTop_ = localStackTop[producerId];
            curStackTop_ = __shfl_sync(0xFFFFFFFFu, curStackTop_, 0);
            unsigned room = (stackCap > curStackTop_) ? (stackCap - curStackTop_) : 0;
            unsigned maxPop = room / 7;
            unsigned popCount = min(min(curStackTop_, (unsigned)GpuConfig::warpSize), maxPop);
            if (lane < popCount)
            {
                unsigned readIdx = curStackTop_ - 1 - lane;
                // assert(readIdx < stackCap && "Producer stack pop: readIdx OOB");
                a = localStackA[producerId][readIdx];
                b = localStackB[producerId][readIdx];
            }
            if (lane == 0) localStackTop[producerId] -= popCount;
            __syncwarp();

            // ── Optimization 1: Flattened child expansion across the warp ──
            // Instead of popCount threads each serially iterating 8 children,
            // store parent data in smem and process ALL children in warp-parallel
            // rounds of 32, reducing atomics from 16 to 2-4 per outer iteration.
            {
                TreeNodeIndex aCO_ = 0, bCO_ = 0;
                bool splitA_ = false;
                bool hasChildren = false;
                bool constSideLeaf_ = false;

                if (lane < popCount)
                {
                    aCO_ = childOffsetLoad(childOffsets, a);
                    bCO_ = childOffsetLoad(childOffsets, b);
                    bool aIsLeaf = (aCO_ == 0);
                    bool bIsLeaf = (bCO_ == 0);
                    splitA_ = (a < b && !aIsLeaf) || bIsLeaf;
                    hasChildren = splitA_ ? !aIsLeaf : !bIsLeaf;
                    constSideLeaf_ = splitA_ ? bIsLeaf : aIsLeaf;
                }

                // Compact parents with children into contiguous smem slots
                unsigned hasMask = __ballot_sync(0xFFFFFFFFu, hasChildren);
                unsigned compIdx = __popc(hasMask & ((1u << lane) - 1));
                unsigned numActiveParents = __popc(hasMask);

                if (hasChildren)
                {
                    smem_parentACO[producerId][compIdx]    = aCO_;
                    smem_parentBCO[producerId][compIdx]    = bCO_;
                    smem_parentA[producerId][compIdx]      = a;
                    smem_parentB[producerId][compIdx]      = b;
                    smem_parentSplitA[producerId][compIdx] = splitA_ ? 1u : 0u;
                    smem_parentConstLeaf[producerId][compIdx] = constSideLeaf_ ? 1u : 0u;
                }
                __syncwarp();

                unsigned totalChildren = numActiveParents * 8;
                unsigned numRounds = (totalChildren + GpuConfig::warpSize - 1)
                                   / GpuConfig::warpSize;

                for (unsigned round = 0; round < numRounds; ++round)
                {
                    unsigned flatIdx = round * GpuConfig::warpSize + lane;
                    bool active = flatIdx < totalChildren;

                    TreeNodeIndex childA = 0, childB = 0;
                    bool cont = false;
                    int isTraversal = 0;

                    if (active)
                    {
                        unsigned pIdx = flatIdx >> 3;   // / 8
                        unsigned cOff = flatIdx & 7;    // % 8
                        bool pSplitA = smem_parentSplitA[producerId][pIdx];
                        childA = pSplitA ? smem_parentACO[producerId][pIdx] + (TreeNodeIndex)cOff
                                         : smem_parentA[producerId][pIdx];
                        childB = pSplitA ? smem_parentB[producerId][pIdx]
                                         : smem_parentBCO[producerId][pIdx] + (TreeNodeIndex)cOff;
                        cont = continuation(childA, childB);
                        if (cont)
                        {
                            bool constSideLeaf = (smem_parentConstLeaf[producerId][pIdx] != 0u);
                            bool childAIsLeaf;
                            bool childBIsLeaf;
                            if (pSplitA)
                            {
                                childAIsLeaf = (childOffsetLoad(childOffsets, childA) == 0);
                                childBIsLeaf = constSideLeaf;
                            }
                            else
                            {
                                childAIsLeaf = constSideLeaf;
                                childBIsLeaf = (childOffsetLoad(childOffsets, childB) == 0);
                            }
                            isTraversal = !(childAIsLeaf && childBIsLeaf) ? 1 : 0;
                        }
                    }

                    bool isInteraction = active && !isTraversal;

                    // Warp-wide ballot for parallel classification
                    unsigned travMask = __ballot_sync(0xFFFFFFFFu, isTraversal);
                    unsigned iactMask = __ballot_sync(0xFFFFFFFFu, isInteraction);

                    unsigned travPos   = __popc(travMask & ((1u << lane) - 1));
                    unsigned totalTrav = __popc(travMask);
                    unsigned iactPos   = __popc(iactMask & ((1u << lane) - 1));
                    unsigned totalIact = __popc(iactMask);

                    unsigned curStackTop, curTmpCount;
                    if (lane == 0)
                    {
                        curStackTop = localStackTop[producerId];
                        curTmpCount = tmpIBCount[producerId];
                        localStackTop[producerId] = curStackTop + totalTrav;
                        tmpIBCount[producerId] = curTmpCount + totalIact;
                    }
                    curStackTop = __shfl_sync(0xFFFFFFFFu, curStackTop, 0);
                    curTmpCount = __shfl_sync(0xFFFFFFFFu, curTmpCount, 0);

                    if (isTraversal)
                    {
                        localStackA[producerId][curStackTop + travPos] = childA;
                        localStackB[producerId][curStackTop + travPos] = childB;
                    }
                    else if (isInteraction)
                    {
                        tmpIBA[producerId][curTmpCount + iactPos]   = childA;
                        tmpIBB[producerId][curTmpCount + iactPos]   = childB;
                        tmpIBP2P[producerId][curTmpCount + iactPos] = cont ? 1 : 0;
                    }

                    // Optimization 2: opportunistic mid-expansion drain to IB.
                    // Push tmpIB items into the main IB between rounds so
                    // consumers get work sooner and the post-expansion drain
                    // loop completes faster (or is skipped entirely).
                    // if (round < numRounds - 1)
                    // {
                    //     unsigned curTmpIB;
                    //     if (tid == 0) curTmpIB = atomicAdd(&tmpIBCount, 0u);
                    //     curTmpIB = __shfl_sync(0xFFFFFFFFu, curTmpIB, 0);

                    //     if (curTmpIB >= GpuConfig::warpSize)
                    //     {
                    //         if (ibStackPush<stackCap>(
                    //                 interactionBufferA, interactionBufferB,
                    //                 interactionBufferIsP2P,
                    //                 &interactionBufferCount, &ibFlag,
                    //                 tmpIBA, tmpIBB, tmpIBP2P, curTmpIB))
                    //         {
                    //             if (tid == 0) atomicExch(&tmpIBCount, 0u);
                    //         }
                    //     }
                    // }
                }
            }
            __syncwarp(); // ensure all writes to tmpIB/localStack are visible

            // __producerIterCount++;
            // if (lane == 0 && producerId == 0
            //     && (__producerIterCount & 0x3FF) == 0
            //     && __producerIterCount > 0)
            //     printf("[P blk=%u] iter=%u lstk=%u tmpIB=%u ibCnt=%u prod=%u tW=%u tR=%u\n",
            //            blockIdx.x, __producerIterCount, localStackTop[producerId],
            //            tmpIBCount[producerId],
            //            atomicAdd(&interactionBufferCount, 0u),
            //            atomicAdd(numActiveProducers, 0u),
            //            atomicAdd(globalTraversalQueue.writeHead, 0u),
            //            atomicAdd(globalTraversalQueue.readHead, 0u));

            // unsigned __drainIter = 0;
            for (;;) {
                unsigned ibCount, curTmpIB, lstop;
                if (lane == 0) {
                    curTmpIB = tmpIBCount[producerId];
                    ibCount  = atomicAdd(&interactionBufferCount, 0u);
                    lstop    = localStackTop[producerId];
                }
                curTmpIB = __shfl_sync(0xFFFFFFFFu, curTmpIB, 0);
                ibCount  = __shfl_sync(0xFFFFFFFFu, ibCount, 0);
                lstop    = __shfl_sync(0xFFFFFFFFu, lstop, 0);
                if (curTmpIB == 0) break;

                if (ibCount > TravConfig::forcePush || (ibCount + curTmpIB > TravConfig::stackCap)) {
                    if (tryPushToGlobal<TravConfig>(globalQueue, tmpIBA[producerId], tmpIBB[producerId], tmpIBP2P[producerId], &tmpIBCount[producerId], true)) {}
                    else if (lstop > TravConfig::travAttemptPush) {
                        tryPushTraversalToGlobal<TravConfig>(globalTraversalQueue, localStackA[producerId], localStackB[producerId], &localStackTop[producerId], false);
                    }
                } else if (ibCount > TravConfig::attemptPush) {
                    if (tryPushToGlobal<TravConfig>(globalQueue, tmpIBA[producerId], tmpIBB[producerId], tmpIBP2P[producerId], &tmpIBCount[producerId], false)) {}
                    else {
                        if (ibQueuePush<TravConfig::stackCap>(interactionBufferA, interactionBufferB, interactionBufferIsP2P,
                                                              &interactionBufferHead, &interactionBufferTail,
                                                              &interactionBufferCount, &ibFlag,
                                                              tmpIBA[producerId], tmpIBB[producerId], tmpIBP2P[producerId], curTmpIB)) {
                            if (lane == 0) tmpIBCount[producerId] = 0;
                        }
                    }
                } else {
                    if (ibQueuePush<TravConfig::stackCap>(interactionBufferA, interactionBufferB, interactionBufferIsP2P,
                                                          &interactionBufferHead, &interactionBufferTail,
                                                          &interactionBufferCount, &ibFlag,
                                                          tmpIBA[producerId], tmpIBB[producerId], tmpIBP2P[producerId], curTmpIB)) {
                        if (lane == 0) tmpIBCount[producerId] = 0;
                    }
                    else if (lstop > TravConfig::travForcePush) {
                        tryPushTraversalToGlobal<TravConfig>(globalTraversalQueue, localStackA[producerId], localStackB[producerId], &localStackTop[producerId], false);
                    }
                }
              
                // -------------------------- DEBUGGING INFO --------------------------
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
            // int iter = 0;
            // unsigned pre = 2;

            bool terminate = false;
            // unsigned wHead = 0, rHead = 0;

            unsigned curLST;
            if (lane == 0) curLST = localStackTop[producerId];
            curLST = __shfl_sync(0xFFFFFFFFu, curLST, 0);
            if (curLST >= TravConfig::travForcePush || (curLST + GpuConfig::warpSize * 8 >= stackCap)) {
                tryPushTraversalToGlobal<TravConfig>(globalTraversalQueue, localStackA[producerId], localStackB[producerId], &localStackTop[producerId], true);
            } else if (curLST > TravConfig::travAttemptPush) {
                tryPushTraversalToGlobal<TravConfig>(globalTraversalQueue, localStackA[producerId], localStackB[producerId], &localStackTop[producerId], false);
            } else if (curLST == 0) {
                // unsigned readLocal
                // do {
                    if (tryPopTraversalFromGlobal<TravConfig>(globalTraversalQueue, localStackA[producerId], localStackB[producerId], &localStackTop[producerId], true)) {
                        if (lane == 0 && !currentlySignaledAsActive) {
                            atomicAdd(numActiveProducers, 1u);
                            __threadfence(); 
                            // printf("[INCR blk=%u] pop from global, pre=%u\n", blockIdx.x, atomicAdd(numActiveProducers, 0u));
                        }
                        if (!currentlySignaledAsActive) currentlySignaledAsActive = true;
                        // break;
                    }
                    else {
                        if (lane == 0 && currentlySignaledAsActive) {
                            atomicSub(numActiveProducers, 1u);
                            __threadfence();
                            // printf("[DECR blk=%u] failed pop from global, pre=%u\n", blockIdx.x, atomicAdd(numActiveProducers, 0u));
                        }
                        // pre = tile.shfl(pre, 0);
                        if (currentlySignaledAsActive) currentlySignaledAsActive = false;
                    }

                    
                // } while (localStackTop == 0 && !terminate);
            } else if (curLST <= TravConfig::travAttemptPop) {
                tryPopTraversalFromGlobal<TravConfig>(globalTraversalQueue, localStackA[producerId], localStackB[producerId], &localStackTop[producerId], false);
            }
           
            __syncwarp();

            if (lane == 0 && !currentlySignaledAsActive) {
                unsigned localCount = localStackTop[producerId];
                unsigned producers = pRef.load(cuda::memory_order_acquire);
                unsigned wHead     = wRef.load(cuda::memory_order_acquire);
                unsigned rHead     = rRef.load(cuda::memory_order_acquire);
                terminate = (localCount == 0 && wHead <= rHead && producers == 0);
                // printf("[TERMINATE CHECK blk=%u] producers=%u\n", blockIdx.x, atomicAdd(numActiveProducers, 0u));
                if (currentlySignaledAsActive && terminate) {
                    atomicSub(numActiveProducers, 1u);
                    __threadfence();
                }
            }
            terminate = __shfl_sync(0xFFFFFFFFu, terminate, 0);
            // if (lane == 0) {
            //     terminate = (pRef.load(cuda::memory_order_acquire) == 0);
            //     printf("[TERMINATE CHECK blk=%u] producers=%u\n", blockIdx.x, atomicAdd(numActiveProducers, 0u));
            // }
            // terminate = __shfl_sync(0xFFFFFFFFu, terminate, 0);
            if (terminate) break;

        }
        // Per-block exit diagnostic
        // if (lane == 0 && producerId == 0)
        // {
        //     bool wasTimeout = (__producerIterCount >= __producerMaxIter);
        //     printf("[P EXIT blk=%u] iters=%u %s\n",
        //            blockIdx.x, __producerIterCount,
        //            wasTimeout ? "TIMEOUT" : "cooperative");
        // }
        // -------------------------- DEBUG INFO --------------------------
        // if (tid == 0) printf("[PRODUCER blk=%u] exiting main loop\n", blockIdx.x);
        // __threadfence(); // ensure all IB/global-queue writes visible before signaling done
        // if(lane == 0)
        // {
        //     unsigned pre = atomicSub(numActiveProducers, 1u);
        //     // printf("[DECR blk=%u] END-OF-LOOP, pre=%u\n", blockIdx.x, pre);
        // }
    }

    // constexpr unsigned __consumerMaxIter = 10000000; // DEBUG: force-exit after this many iterations
    // unsigned consumerSpinCount = 0;
    auto tile = cg::coalesced_threads();
    auto wRef = cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*globalQueue.writeHead);
    auto rRef = cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*globalQueue.readHead);
    const unsigned refillWarpId = (numWarps > producerWarpCount) ? producerWarpCount : 0u;
    for (;;)
    {
        // if (consumerSpinCount >= __consumerMaxIter)
        // {
        //     if (lane == 0 && blockIdx.x == 0 && warpId == producerWarpCount)
        //         printf("[C blk=0] TIMEOUT after %u spins, ibCnt=%u prod=%u iW=%u iR=%u\n",
        //                consumerSpinCount,
        //                atomicAdd(&interactionBufferCount, 0u),
        //                atomicAdd(numActiveProducers, 0u),
        //                atomicAdd(globalQueue.writeHead, 0u),
        //                atomicAdd(globalQueue.readHead, 0u));
        //     break;
        // }
        // consumerSpinCount++;
        // if (lane == 0 && blockIdx.x == 0 && warpId == producerWarpCount
        //     && (consumerSpinCount & 0xFFFFF) == 0)
        //     printf("[C blk=0] spin=%u ibCnt=%u ibFlg=%u gPop=%u "
        //            "iW=%u iR=%u tW=%u tR=%u prod=%u lst0=%u\n",
        //            consumerSpinCount,
        //            atomicAdd(&interactionBufferCount, 0u), atomicAdd(&ibFlag, 0u),
        //            atomicAdd(&globalPopFlag, 0u),
        //            atomicAdd(globalQueue.writeHead, 0u),
        //            atomicAdd(globalQueue.readHead, 0u),
        //            atomicAdd(globalTraversalQueue.writeHead, 0u),
        //            atomicAdd(globalTraversalQueue.readHead, 0u),
        //            atomicAdd(numActiveProducers, 0u),
        //            atomicAdd(&localStackTop[0], 0u));

        // ── Try to pop items from local IB (flag-guarded) ──
        TreeNodeIndex itemA = 0u, itemB = 0u;
        int itemIsP2P = false;
        unsigned popCount = 0u;

        unsigned numLocal;
        if (lane == 0) {
            numLocal = atomicAdd(&interactionBufferCount, 0u);
        }
        numLocal = tile.shfl(numLocal, 0);

        if (numLocal < TravConfig::attemptPop && warpId == refillWarpId) {
            bool success = false;
            if (lane == 0) success = acquireFlag(&globalPopFlag);
            success = tile.shfl(success, 0);

            if(success) {
                bool force = numLocal <= TravConfig::forcePop;
                tryPopFromGlobal<TravConfig>(globalQueue,
                                             interactionBufferA, interactionBufferB, interactionBufferIsP2P,
                                             &interactionBufferHead, &interactionBufferTail,
                                             &interactionBufferCount,
                                             force, &ibFlag);
                if (lane == 0) releaseFlag(&globalPopFlag);
            }
        }

        // __syncwarp();

        if (lane == 0) {
            numLocal = atomicAdd(&interactionBufferCount, 0u);
        }
        numLocal = tile.shfl(numLocal, 0);

        if (numLocal > 0) {
            ibQueuePop<TravConfig::stackCap>(interactionBufferA, interactionBufferB, interactionBufferIsP2P,
                                             &interactionBufferHead, &interactionBufferTail,
                                             &interactionBufferCount, &ibFlag,
                                             itemA, itemB, itemIsP2P, popCount);
        }

        // __syncwarp();

        // ── Process popped items ──
        if (lane < popCount)
        {
            // if (!(itemIsP2P == 0 || itemIsP2P == 1)) {printf("Item: %d\n", itemIsP2P);}
            // assert((itemIsP2P == 0 || itemIsP2P == 1) && "Consumer: itemIsP2P corrupted");
            // assert(itemA >= 0 && "Consumer: itemA is negative");
            // assert(itemB >= 0 && "Consumer: itemB is negative");
            if (itemIsP2P)  p2p(itemA, itemB);
            else            m2l(itemA, itemB);
        }
        __syncwarp();

        // unsigned localCount, wHead, rHead, producers;

        bool terminate = false;
        if (popCount == 0) {
            if (lane == 0) {
                unsigned localCount = atomicAdd(&interactionBufferCount, 0u);  // force reload from smem
                unsigned producers = pRef.load(cuda::memory_order_acquire);
                unsigned wHead     = wRef.load(cuda::memory_order_acquire);
                unsigned rHead     = rRef.load(cuda::memory_order_acquire);
                terminate = (localCount == 0 && wHead <= rHead && producers == 0);
            }
            terminate = tile.shfl(terminate, 0);
        }
        if (terminate) break;
    }

    // ----- Per-block exit diagnostic -----
    // if (lane == 0)
    // {
    //     bool wasTimeout = (consumerSpinCount >= __consumerMaxIter);
    //     printf("[C EXIT blk=%u] spin=%u %s\n",
    //            blockIdx.x, consumerSpinCount,
    //            wasTimeout ? "TIMEOUT" : "cooperative");
    // }

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
        assignPairByFrontierDeterministic(childOffsets, a, b,
                                          block_in_cluster, blocksPerCluster,
                                          active_blocks,
                                          std::forward<MAC>(continuation));

    // non-producer blocks decrement active count immediately
    if (!producer && cg::this_thread_block().thread_rank() == 0)
    {
        unsigned pre = atomicSub(numActiveProducers, kProducerWarpsPerBlock);
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
    // Each block registers its producer warps on arrival. Blocks that
    // are never scheduled simply never increment, preventing the
    // occupancy deadlock when totalBlocks > GPU co-resident capacity.
    if (cg::this_thread_block().thread_rank() == 0)
    {
        atomicAdd(numActiveProducers, kProducerWarpsPerBlock);
    }
    cg::this_thread_block().sync();

    const unsigned cluster_id  = cluster_rank_in_grid();
    const unsigned numClusters = num_clusters_runtime();

    TreeNodeIndex a = rootA, b = rootB;

    unsigned active_clusters = 0;
    const bool active_cluster =
        assignPairByFrontierDeterministic(childOffsets, a, b,
                                          cluster_id, numClusters,
                                          active_clusters,
                                          std::forward<MAC>(continuation));

    if (!active_cluster)
    {
        if (cg::this_thread_block().thread_rank() == 0)
        {
            unsigned pre = atomicSub(numActiveProducers, kProducerWarpsPerBlock);
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
