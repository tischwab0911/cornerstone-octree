/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cstone
{

namespace queue_detail
{

__device__ __forceinline__ int acquireFlag(unsigned* flag)
{
    return atomicCAS(flag, 0u, 1u) == 0u ? 1 : 0;
}

__device__ __forceinline__ void releaseFlag(unsigned* flag)
{
    __threadfence_block();
    atomicExch(flag, 0u);
}

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

} // namespace queue_detail

template<class TreeNodeIndex>
struct InteractionQueue
{
    TreeNodeIndex* nodeA{nullptr};    //!< [capacity] interaction node A
    TreeNodeIndex* nodeB{nullptr};    //!< [capacity] interaction node B
    int*           isP2P{nullptr};    //!< [capacity] 1=p2p, 0=m2l (int for alignment)
    unsigned*      writeHead{nullptr};//!< atomic monotonic segment write counter
    unsigned*      readHead{nullptr}; //!< atomic monotonic segment read counter
    unsigned*      segCount{nullptr}; //!< [numSegments] valid item count per segment
    unsigned*      segReady{nullptr}; //!< [numSegments] per-segment ready flags
    unsigned       numSegments{0};
    unsigned       segmentSize{0};

    __host__ __device__ constexpr InteractionQueue() = default;

    __host__ __device__ constexpr InteractionQueue(TreeNodeIndex* nodeA_,
                                                   TreeNodeIndex* nodeB_,
                                                   int*           isP2P_,
                                                   unsigned*      writeHead_,
                                                   unsigned*      readHead_,
                                                   unsigned*      segCount_,
                                                   unsigned*      segReady_,
                                                   unsigned       numSegments_,
                                                   unsigned       segmentSize_)
        : nodeA(nodeA_)
        , nodeB(nodeB_)
        , isP2P(isP2P_)
        , writeHead(writeHead_)
        , readHead(readHead_)
        , segCount(segCount_)
        , segReady(segReady_)
        , numSegments(numSegments_)
        , segmentSize(segmentSize_)
    {}

    __device__ __forceinline__ bool push(TreeNodeIndex* __restrict__ bufA,
                                         TreeNodeIndex* __restrict__ bufB,
                                         int*           __restrict__ bufIsP2P,
                                         unsigned*      __restrict__ bufCount,
                                         bool                        force) const
    {
        auto tile           = cooperative_groups::coalesced_threads();
        const unsigned lane = tile.thread_rank();
        const unsigned chunk = segmentSize;

        unsigned wHead = 0;
        unsigned rHead = 0;
        bool claimed = false;
        do {
            if (lane == 0)
            {
                wHead = *writeHead;
                rHead = *readHead;
            }
            wHead = tile.shfl(wHead, 0);
            rHead = tile.shfl(rHead, 0);

            if (wHead - rHead >= numSegments) return false;

            if (lane == 0) claimed = (atomicCAS(writeHead, wHead, wHead + 1) == wHead);
            claimed = tile.shfl(claimed, 0);
        } while (force && !claimed);

        if (!claimed) return false;

        const unsigned seg = wHead % numSegments;
        if (lane == 0)
        {
            for (unsigned spin = 0;; ++spin)
            {
                if (atomicAdd(&segReady[seg], 0u) == 0u) break;
                queue_detail::spinBackoff(spin);
            }
        }
        tile.sync();

        const unsigned base = seg * chunk;

        unsigned count = 0;
        if (lane == 0) count = *bufCount;
        count = tile.shfl(count, 0);

        const unsigned numPushed = min(chunk, count);
        const unsigned srcStart  = count - numPushed;

        for (unsigned i = lane; i < numPushed; i += tile.num_threads())
        {
            nodeA[base + i] = bufA[srcStart + i];
            nodeB[base + i] = bufB[srcStart + i];
            isP2P[base + i] = bufIsP2P[srcStart + i];
        }

        __threadfence();
        if (lane == 0)
        {
            *bufCount      = count - numPushed;
            segCount[seg]  = numPushed;
            __threadfence();
            atomicExch(&segReady[seg], 1u);
        }
        return true;
    }

    __device__ __forceinline__ bool pop(TreeNodeIndex* __restrict__ bufA,
                                        TreeNodeIndex* __restrict__ bufB,
                                        int*           __restrict__ bufIsP2P,
                                        unsigned*      __restrict__ bufHead,
                                        unsigned*      __restrict__ bufTail,
                                        unsigned*      __restrict__ bufCount,
                                        unsigned                    localCapacity,
                                        bool                        force,
                                        unsigned*      __restrict__ ibFlag) const
    {
        auto tile           = cooperative_groups::coalesced_threads();
        const unsigned lane = tile.thread_rank();
        const unsigned chunk = segmentSize;

        unsigned rHead = 0;
        unsigned wHead = 0;
        bool claimed = false;
        do {
            if (lane == 0)
            {
                rHead = *readHead;
                wHead = *writeHead;
            }
            rHead = tile.shfl(rHead, 0);
            wHead = tile.shfl(wHead, 0);

            if (wHead <= rHead) return false;

            if (lane == 0) claimed = (atomicCAS(readHead, rHead, rHead + 1) == rHead);
            claimed = tile.shfl(claimed, 0);
        } while (force && !claimed);

        if (!claimed) return false;

        const unsigned seg = rHead % numSegments;
        if (lane == 0)
        {
            for (unsigned spin = 0;; ++spin)
            {
                if (atomicAdd(&segReady[seg], 0u) == 1u) break;
                queue_detail::spinBackoff(spin);
            }
        }
        tile.sync();
        __threadfence();

        const unsigned base = seg * chunk;

        unsigned validCount = 0;
        if (lane == 0) validCount = segCount[seg];
        validCount = tile.shfl(validCount, 0);

        unsigned tail = 0;
        unsigned count = 0;
        bool haveSpace = false;
        while (!haveSpace)
        {
            bool success = false;
            if (lane == 0) success = queue_detail::acquireFlag(ibFlag);
            success = tile.shfl(success, 0);
            if (!success)
            {
                if (lane == 0) queue_detail::spinBackoff(8);
                continue;
            }

            if (lane == 0)
            {
                tail      = *bufTail;
                count     = *bufCount;
                haveSpace = (count + validCount <= localCapacity);
                if (!haveSpace) queue_detail::releaseFlag(ibFlag);
                if (!haveSpace) queue_detail::spinBackoff(8);
            }
            tail      = tile.shfl(tail, 0);
            count     = tile.shfl(count, 0);
            haveSpace = tile.shfl(haveSpace, 0);
        }

        for (unsigned j = lane; j < validCount; j += tile.num_threads())
        {
            unsigned writeIdx = tail + j;
            if (writeIdx >= localCapacity) writeIdx -= localCapacity;
            bufA[writeIdx]     = nodeA[base + j];
            bufB[writeIdx]     = nodeB[base + j];
            bufIsP2P[writeIdx] = isP2P[base + j];
        }

        __threadfence_block();
        if (lane == 0)
        {
            unsigned newTail = tail + validCount;
            if (newTail >= localCapacity) newTail -= localCapacity;
            *bufTail  = newTail;
            *bufCount = count + validCount;
            (void)bufHead;
            queue_detail::releaseFlag(ibFlag);
            segCount[seg] = 0u;
            __threadfence();
            atomicExch(&segReady[seg], 0u);
        }
        return true;
    }
};

template<class TreeNodeIndex>
struct TraversalQueue
{
    TreeNodeIndex* nodeA{nullptr};    //!< [capacity] traversal node A
    TreeNodeIndex* nodeB{nullptr};    //!< [capacity] traversal node B
    unsigned*      writeHead{nullptr};//!< atomic monotonic segment write counter
    unsigned*      readHead{nullptr}; //!< atomic monotonic segment read counter
    unsigned*      segReady{nullptr}; //!< [numSegments] per-segment ready flags
    unsigned       numSegments{0};
    unsigned       segmentSize{0};

    __host__ __device__ constexpr TraversalQueue() = default;

    __host__ __device__ constexpr TraversalQueue(TreeNodeIndex* nodeA_,
                                                 TreeNodeIndex* nodeB_,
                                                 unsigned*      writeHead_,
                                                 unsigned*      readHead_,
                                                 unsigned*      segReady_,
                                                 unsigned       numSegments_,
                                                 unsigned       segmentSize_)
        : nodeA(nodeA_)
        , nodeB(nodeB_)
        , writeHead(writeHead_)
        , readHead(readHead_)
        , segReady(segReady_)
        , numSegments(numSegments_)
        , segmentSize(segmentSize_)
    {}

    __device__ __forceinline__ bool push(TreeNodeIndex* __restrict__ bufA,
                                         TreeNodeIndex* __restrict__ bufB,
                                         unsigned*      __restrict__ bufCount,
                                         bool                        force) const
    {
        auto tile           = cooperative_groups::coalesced_threads();
        const unsigned lane = tile.thread_rank();
        const unsigned chunk = segmentSize;

        bool claimed = false;
        unsigned rHead = 0;
        unsigned wHead = 0;
        do {
            if (lane == 0)
            {
                wHead = *writeHead;
                rHead = *readHead;
            }
            wHead = tile.shfl(wHead, 0);
            rHead = tile.shfl(rHead, 0);

            if (wHead - rHead >= numSegments) return false;

            if (lane == 0) claimed = (atomicCAS(writeHead, wHead, wHead + 1) == wHead);
            claimed = tile.shfl(claimed, 0);
        } while (force && !claimed);

        if (!claimed) return false;

        const unsigned seg = wHead % numSegments;
        if (lane == 0)
        {
            for (unsigned spin = 0;; ++spin)
            {
                if (atomicAdd(&segReady[seg], 0u) == 0u) break;
                queue_detail::spinBackoff(spin);
            }
        }
        tile.sync();

        const unsigned base = seg * chunk;
        const unsigned count = *bufCount;
        const unsigned numPushed = min(chunk, count);
        for (unsigned i = lane; i < numPushed; i += tile.num_threads())
        {
            const unsigned srcIdx = count - numPushed + i;
            nodeA[base + i] = bufA[srcIdx];
            nodeB[base + i] = bufB[srcIdx];
        }

        __threadfence();
        if (lane == 0)
        {
            *bufCount = count - numPushed;
            __threadfence();
            atomicExch(&segReady[seg], 1u);
        }
        return true;
    }

    __device__ __forceinline__ bool pop(TreeNodeIndex* __restrict__ bufA,
                                        TreeNodeIndex* __restrict__ bufB,
                                        unsigned*      __restrict__ bufCount,
                                        bool                        force) const
    {
        auto tile           = cooperative_groups::coalesced_threads();
        const unsigned lane = tile.thread_rank();
        const unsigned chunk = segmentSize;

        unsigned rHead = 0;
        unsigned wHead = 0;
        bool claimed = false;
        do {
            if (lane == 0)
            {
                rHead = *readHead;
                wHead = *writeHead;
            }
            rHead = tile.shfl(rHead, 0);
            wHead = tile.shfl(wHead, 0);

            if (wHead <= rHead) return false;

            if (lane == 0) claimed = (atomicCAS(readHead, rHead, rHead + 1) == rHead);
            claimed = tile.shfl(claimed, 0);
        } while (force && !claimed);

        if (!claimed) return false;

        const unsigned seg = rHead % numSegments;
        if (lane == 0)
        {
            for (unsigned spin = 0;; ++spin)
            {
                if (atomicAdd(&segReady[seg], 0u) == 1u) break;
                queue_detail::spinBackoff(spin);
            }
        }
        tile.sync();
        __threadfence();

        const unsigned base = seg * chunk;
        const unsigned count = *bufCount;
        for (unsigned i = lane; i < chunk; i += tile.num_threads())
        {
            bufA[count + i] = nodeA[base + i];
            bufB[count + i] = nodeB[base + i];
        }

        if (lane == 0)
        {
            *bufCount = count + chunk;
            __threadfence();
            atomicExch(&segReady[seg], 0u);
        }
        return true;
    }
};

} // namespace cstone
