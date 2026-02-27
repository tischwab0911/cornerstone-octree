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
 * Dual tree traversal on the GPU is the basis for many algorithms
 */

#pragma once
#pragma nv_diag_suppress static_var_with_dynamic_init

#include "cstone/tree/octree.hpp"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/warpscan.cuh"
#include <cuda/atomic>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace cg = cooperative_groups;

// using ClusterFlag = cuda::atomic<int, cg::thread_scope_cluster>;

__device__ __forceinline__ unsigned linear_block_rank()
{
    cg::grid_group grid = cg::this_grid();
    return grid.block_rank(); // linear block index across the grid
}

__device__ __forceinline__ unsigned blocks_per_cluster_runtime()
{
    cg::cluster_group cl = cg::this_cluster();
    // For 1D clusters, num_blocks() gives blocks in the cluster.
    return cl.num_blocks();
}

__device__ __forceinline__ unsigned block_rank_in_cluster()
{
    cg::cluster_group cl = cg::this_cluster();
    return cl.block_rank(); // 0..blocks_per_cluster-1
}

__device__ __forceinline__ unsigned cluster_rank_in_grid()
{
    const unsigned bpc = blocks_per_cluster_runtime();
    return linear_block_rank() / bpc;
}

__device__ __forceinline__ unsigned num_clusters_runtime()
{
    const unsigned bpc = blocks_per_cluster_runtime();
    return cg::this_grid().num_blocks() / bpc;
}

__device__ __forceinline__ unsigned thread_rank_in_block()
{
    cg::thread_block tb = cg::this_thread_block();
    return tb.thread_rank();
}

__device__ __forceinline__ unsigned lane_in_tile()
{
    auto tile = cg::coalesced_threads();
    return tile.thread_rank();
}

__device__ __forceinline__ unsigned num_lanes_in_tile()
{
    auto tile = cg::coalesced_threads();
    return tile.size();
}

HOST_DEVICE_FUN __forceinline__
constexpr std::size_t align_up(std::size_t x, std::size_t a)
{
    // a must be a power of two for this form
    return (x + (a - 1)) & ~(a - 1);
}


namespace cstone
{

static constexpr int kDSMCapacity = GpuConfig::warpSize * 16;

struct DSMStack {
    int nextFreePos;
    TreeNodeIndex nodeAStack[kDSMCapacity];
    TreeNodeIndex nodeBStack[kDSMCapacity];
};

__device__ __forceinline__ int acquireFlag(int* flag) {
    return !(atomicCAS(flag, 0, 1));
}

__device__ __forceinline__ void releaseFlag(int* flag) {
    atomicExch(flag, 0);
}

__device__ inline bool attemptPush(int* flag, DSMStack* dsmStack,
                            TreeNodeIndex* nodeABuffer, TreeNodeIndex* nodeBBuffer,
                            int producedPairs)
{
    auto tile = cg::coalesced_threads();
    const unsigned lane = tile.thread_rank();
    const unsigned tileSz  = tile.size();

    int success;
    if (lane == 0) success = acquireFlag(flag);
    success = shflSync(success, 0);
    if (!success) return false;

    int nextFreePos;
    if (lane == 0) nextFreePos = dsmStack->nextFreePos;
    nextFreePos = shflSync(nextFreePos, 0);

    if (__builtin_expect(nextFreePos + (int)tileSz > kDSMCapacity, 0)) {
        if (lane == 0) releaseFlag(flag);
        return false;
    }

    // if (nextFreePos < 0 || nextFreePos > kDSMCapacity) {
    //     if (lane == 0)
    //         printf("[attemptPush] corrupt DSM nextFreePos=%d (capacity=%d)\n",
    //                nextFreePos, kDSMCapacity);
    //     assert(false);
    // }

    int offset = cg::exclusive_scan(tile, producedPairs, cg::plus<int>());
    int startIdx = nextFreePos + offset;

    if (__builtin_expect(startIdx + producedPairs > kDSMCapacity, 0)) {
        if (lane == 0) {
            printf("[attemptPush] DSM overflow: lane %u writing [%d, %d), capacity=%d\n",
                   lane, startIdx, startIdx + producedPairs, kDSMCapacity);
            releaseFlag(flag);
        }
        return false;
    }
    for (int i = 0; i < producedPairs; ++i) {
        dsmStack->nodeAStack[startIdx + i] = nodeABuffer[i];
        dsmStack->nodeBStack[startIdx + i] = nodeBBuffer[i];
    }

    if (lane == num_lanes_in_tile()-1) {
        dsmStack->nextFreePos = startIdx+producedPairs;
        releaseFlag(flag);
    }
    
    return true;
}

__device__ inline unsigned attemptPop(int* flag, DSMStack* dsmStack,
                                      TreeNodeIndex* nodeABuffer, TreeNodeIndex* nodeBBuffer)
{
    auto tile = cg::coalesced_threads();
    unsigned lane = tile.thread_rank();

    int success;
    if (lane == 0) success = atomicCAS(flag, 0, 1) == 0;
    success = shflSync(success, 0);
    if (!success) return 0;

    int nextFreePos = 0;
    if (lane == 0) nextFreePos = dsmStack->nextFreePos;
    nextFreePos = shflSync(nextFreePos, 0);

    unsigned n = nextFreePos >= (int)GpuConfig::warpSize ? (unsigned)GpuConfig::warpSize
                                                         : (unsigned)nextFreePos;


    if (lane < n) {
        unsigned readFromPos = (unsigned)nextFreePos - n + lane;
        nodeABuffer[0] = dsmStack->nodeAStack[readFromPos];
        nodeBBuffer[0] = dsmStack->nodeBStack[readFromPos];
    }

    if (lane == 0) {
        dsmStack->nextFreePos = nextFreePos - (int)n;
        atomicExch(flag, 0);
    }
    return n;
}

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

using NodePair = util::array<TreeNodeIndex, 2>;

template<int Stages, int numConsumersPerBlock, class MAC, class M2L, class P2P>
__device__ void dualTraversalBlock(
    const TreeNodeIndex* __restrict__ childOffsets,
    TreeNodeIndex a, TreeNodeIndex b,
    DSMStack* __restrict__ dsmStack,
    int*      __restrict__ dsmFlag,
    int*      __restrict__ authActiveCount,
    bool hasInitialWork,
    MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    using State = cuda::pipeline_shared_state<cuda::thread_scope_block, Stages>;
    constexpr TreeNodeIndex maxNewChildren = 8;

    cg::thread_block block = cg::this_thread_block();
    cuda::std::size_t blockThreadIdx = block.thread_rank();

    // SoA pipeline buffers, single merged pipeline.
    __shared__ TreeNodeIndex bufferNodeA[Stages][numConsumersPerBlock * GpuConfig::warpSize];
    __shared__ TreeNodeIndex bufferNodeB[Stages][numConsumersPerBlock * GpuConfig::warpSize];
    __shared__ uint32_t      bufferAction[Stages][numConsumersPerBlock * GpuConfig::warpSize];
    __shared__ State         ptrPipeline;

    auto thread_role = (blockThreadIdx < GpuConfig::warpSize) ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;
    auto pipeline = cuda::make_pipeline(block, &ptrPipeline, thread_role);

    if (thread_role == cuda::pipeline_role::producer) {

        auto tile = cg::coalesced_threads();
        int lastLaneInTile = tile.size()-1;

        int prodStageIdx = 0;
        int consumerIdx  = 0;

        constexpr int stackSize = GpuConfig::warpSize * maxNewChildren * 4;
        constexpr int kSpillThreshold = stackSize * 1 / 4;
        __shared__ __align__(alignof(TreeNodeIndex)) TreeNodeIndex nodeAStack[stackSize];
        __shared__ __align__(alignof(TreeNodeIndex)) TreeNodeIndex nodeBStack[stackSize];

        // Per-thread buffers for stack push-back
        TreeNodeIndex nodeABuffer[maxNewChildren];
        TreeNodeIndex nodeBBuffer[maxNewChildren];

        // Per-consumer pipeline item buffers
        TreeNodeIndex nodeAItemBuffer[numConsumersPerBlock];
        TreeNodeIndex nodeBItemBuffer[numConsumersPerBlock];
        uint32_t      actionBuffer[numConsumersPerBlock];

        if (hasInitialWork && blockThreadIdx == 0) {
            nodeAStack[0] = a;
            nodeBStack[0] = b;
        }
        int nextFreePos = hasInitialWork ? 1 : 0;
        bool blockIsActive = hasInitialWork;
        bool stackContainsValues = hasInitialWork;

        while (true) {
            // ── Steal-or-terminate when local stack is empty ─────────────────
            if (!stackContainsValues) {
                // Go idle: decrement count (only once per transition active→idle)
                if (blockIsActive) {
                    if (blockThreadIdx == 0) atomicSub(authActiveCount, 1);
                    blockIsActive = false;
                }

                // Try to steal; each lane writes its popped pair to nodeAStack[lane]
                unsigned stolen = attemptPop(dsmFlag, dsmStack,
                                             &nodeAStack[blockThreadIdx],
                                             &nodeBStack[blockThreadIdx]);
                if (stolen > 0) {
                    if (blockThreadIdx == 0) atomicAdd(authActiveCount, 1);
                    blockIsActive = true;
                    nextFreePos = (int)stolen;
                    stackContainsValues = true;
                    // fall through to traversal iteration below
                } else {
                    // DSM empty (or lock contention) — check termination
                    int remain = *authActiveCount;
                    if (remain <= 0) break;  // truly done → exit while(true)
                    continue;                // other blocks active → retry steal
                }
            }

            // ── Traversal iteration ────────────────────────────────────────────
            int stackPos = nextFreePos - (int)GpuConfig::warpSize + (int)blockThreadIdx;
            bool validPos = (stackPos >= 0 && stackPos < stackSize);

            TreeNodeIndex nodeA = validPos ? nodeAStack[stackPos] : 0;
            TreeNodeIndex nodeB = validPos ? nodeBStack[stackPos] : 0;

            // Load child offsets once; derive leaf flags directly — avoids double-read.
            TreeNodeIndex nodeAChildOffset = validPos ? childOffsets[nodeA] : 0;
            TreeNodeIndex nodeBChildOffset = validPos ? childOffsets[nodeB] : 0;
            bool nodeAIsLeaf = (nodeAChildOffset == 0);
            bool nodeBIsLeaf = (nodeBChildOffset == 0);
            TreeNodeIndex divideNodeA = ((nodeA < nodeB && !nodeAIsLeaf) || nodeBIsLeaf) ? 1 : 0;

            int producedPairs = 0;

            #pragma unroll
            for (TreeNodeIndex octant = 0; octant < maxNewChildren; ++octant) {
                TreeNodeIndex nodeAChildIdx = validPos ? (1 - divideNodeA) * nodeA + (nodeAChildOffset + octant) * divideNodeA : 0u;
                TreeNodeIndex nodeBChildIdx = validPos ? divideNodeA * nodeB       + (nodeBChildOffset + octant) * (1 - divideNodeA) : 0u;
                bool continueTraversal = validPos ? continuation(nodeAChildIdx, nodeBChildIdx) : false;

                // The fixed side's leaf flag is constant across all 8 octants:
                // when dividing A, nodeB is fixed -> nodeBChildIsLeaf == nodeBIsLeaf;
                // when dividing B, nodeA is fixed -> nodeAChildIsLeaf == nodeAIsLeaf.
                // Only call isLeaf (global read) for the variable (subdivided) side.
                bool nodeAChildIsLeaf = divideNodeA ? isLeaf(childOffsets, nodeAChildIdx) : nodeAIsLeaf;
                bool nodeBChildIsLeaf = divideNodeA ? nodeBIsLeaf : isLeaf(childOffsets, nodeBChildIdx);

                int addPairToLocalStack = (continueTraversal && !(nodeAChildIsLeaf && nodeBChildIsLeaf)) ? 1 : 0;
                if (addPairToLocalStack) {
                    nodeABuffer[producedPairs] = nodeAChildIdx;
                    nodeBBuffer[producedPairs] = nodeBChildIdx;
                    ++producedPairs;
                }

                // 0=skip, 1=m2l, 2=p2p
                uint32_t action = 0;
                if (!addPairToLocalStack && !continueTraversal && (validPos == 1)) { action = 1; }
                if (!addPairToLocalStack &&  continueTraversal)                    { action = 2; }

                nodeAItemBuffer[consumerIdx] = nodeAChildIdx;
                nodeBItemBuffer[consumerIdx] = nodeBChildIdx;
                actionBuffer[consumerIdx]    = action;
                consumerIdx = (consumerIdx + 1) % numConsumersPerBlock;

                if ((consumerIdx == 0) || (octant == maxNewChildren - 1)) {
                    pipeline.producer_acquire();
                    #pragma unroll
                    for (int consumer = 0; consumer < numConsumersPerBlock; ++consumer) {
                        if ((consumer + numConsumersPerBlock * (octant / numConsumersPerBlock)) >= maxNewChildren) {
                            nodeAItemBuffer[consumer] = 0;
                            nodeBItemBuffer[consumer] = 0;
                            actionBuffer[consumer]    = 0;
                            consumerIdx = 0;
                        }
                        cuda::memcpy_async(&bufferNodeA[prodStageIdx][consumer * GpuConfig::warpSize + blockThreadIdx],
                                           &nodeAItemBuffer[consumer], sizeof(TreeNodeIndex), pipeline);
                        cuda::memcpy_async(&bufferNodeB[prodStageIdx][consumer * GpuConfig::warpSize + blockThreadIdx],
                                           &nodeBItemBuffer[consumer], sizeof(TreeNodeIndex), pipeline);
                        cuda::memcpy_async(&bufferAction[prodStageIdx][consumer * GpuConfig::warpSize + blockThreadIdx],
                                           &actionBuffer[consumer], sizeof(uint32_t), pipeline);
                    }
                    pipeline.producer_commit();
                    prodStageIdx = (prodStageIdx + 1) % Stages;
                }
            }

            const int numPopped = inclusiveScanBool(validPos);
            int numToPush = cg::exclusive_scan(tile, producedPairs, cg::plus<int>());
            tile.sync();

            int baseWritePos = 0;

            if (blockThreadIdx == lastLaneInTile) {
                int totalPopped = numPopped;
                int totalToPush = numToPush + producedPairs;
                baseWritePos = nextFreePos - totalPopped;
                nextFreePos  = baseWritePos + totalToPush;
            }

            nextFreePos  = tile.shfl(nextFreePos, lastLaneInTile);
            baseWritePos = tile.shfl(baseWritePos, lastLaneInTile);

            int writeNewPos = baseWritePos + numToPush;

            // add pairs from buffer to shmem
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i < producedPairs) {
                    // if (writeNewPos + i < 0 || writeNewPos + i >= stackSize) {
                    //     printf("[dualTraversalBlock] local stack OOB write:"
                    //            " writeNewPos=%d i=%d stackSize=%d nextFreePos=%d\n",
                    //            writeNewPos, i, stackSize, nextFreePos);
                    //     assert(false);
                    // }
                    nodeAStack[writeNewPos + i] = nodeABuffer[i];
                    nodeBStack[writeNewPos + i] = nodeBBuffer[i];
                }
            }

            if (nextFreePos == 0) stackContainsValues = false;

            // ── Spill: push top warp-size entries to DSM when stack is ¾ full ──
            if (stackContainsValues
                && nextFreePos >= (int)GpuConfig::warpSize
                && nextFreePos > kSpillThreshold)
            {
                // printf("Attempt Spill!\n");
                int spillIdx = nextFreePos - (int)GpuConfig::warpSize + (int)blockThreadIdx;
                // if (spillIdx < 0 || spillIdx >= stackSize) {
                //     printf("[dualTraversalBlock] spill read OOB:"
                //            " spillIdx=%d (nextFreePos=%d blockThreadIdx=%u stackSize=%d)\n",
                //            spillIdx, nextFreePos, (unsigned)blockThreadIdx, stackSize);
                //     assert(false);
                // }
                TreeNodeIndex spillA = nodeAStack[spillIdx];
                TreeNodeIndex spillB = nodeBStack[spillIdx];
                bool spilled = attemptPush(dsmFlag, dsmStack, &spillA, &spillB, 1);
                if (spilled) { nextFreePos -= (int)GpuConfig::warpSize; }
            }
        }

        // ── Send quit signals to consumers ────────────────────────────────────
        {
            TreeNodeIndex quitNode   = 0;
            uint32_t      quitAction = 3;
            for (int stage = 0; stage < Stages; ++stage) {
                int s = (prodStageIdx + stage) % Stages;
                pipeline.producer_acquire();
                #pragma unroll
                for (int consumer = 0; consumer < numConsumersPerBlock; ++consumer) {
                    cuda::memcpy_async(&bufferNodeA[s][consumer * GpuConfig::warpSize + blockThreadIdx],
                                       &quitNode, sizeof(TreeNodeIndex), pipeline);
                    cuda::memcpy_async(&bufferNodeB[s][consumer * GpuConfig::warpSize + blockThreadIdx],
                                       &quitNode, sizeof(TreeNodeIndex), pipeline);
                    cuda::memcpy_async(&bufferAction[s][consumer * GpuConfig::warpSize + blockThreadIdx],
                                       &quitAction, sizeof(uint32_t), pipeline);
                }
                pipeline.producer_commit();
            }
        }

        pipeline.quit();

    } else if (thread_role == cuda::pipeline_role::consumer) {

        int consStageIdx   = 0;
        int release_signal = 0;
        int tileThreadIdx  = blockThreadIdx - GpuConfig::warpSize;

        while (release_signal < Stages) {
            pipeline.consumer_wait();
            TreeNodeIndex nodeA  = bufferNodeA[consStageIdx][tileThreadIdx];
            TreeNodeIndex nodeB  = bufferNodeB[consStageIdx][tileThreadIdx];
            uint32_t      action = bufferAction[consStageIdx][tileThreadIdx];
            pipeline.consumer_release();

            if      (action == 1) { m2l(nodeA, nodeB); }
            else if (action == 2) { p2p(nodeA, nodeB); }
            else if (action == 3) { ++release_signal; }
            consStageIdx = (consStageIdx + 1) % Stages;
        }
        pipeline.quit();
    }
}

template <int numConsumersPerBlock, class MAC, class M2L, class P2P>
__device__ void dualTraversalTBC(const TreeNodeIndex* __restrict__ childOffsets,
                                 TreeNodeIndex a, TreeNodeIndex b,
                                 MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    __shared__ DSMStack clusterStack;
    __shared__ int clusterActiveCount;
    __shared__ int flag;

    const unsigned block_in_cluster = block_rank_in_cluster();
    const unsigned blocksPerCluster = blocks_per_cluster_runtime();

    assert(blockDim.x % GpuConfig::warpSize == 0);

    unsigned active_blocks = 0;
    const bool active_block =
        assignPairBySplitting_regress(childOffsets, a, b,
                                      block_in_cluster, blocksPerCluster,
                                      active_blocks,
                                      std::forward<MAC>(continuation));

    cg::cluster_group cluster = cg::this_cluster();
    DSMStack* remoteStack     = (DSMStack*)cluster.map_shared_rank(&clusterStack, 0);
    int*      remoteFlag      = (int*)cluster.map_shared_rank(&flag, 0);
    int*      authActiveCount = (int*)cluster.map_shared_rank(&clusterActiveCount, 0);

    if (threadIdx.x == 0 && block_in_cluster == 0) {
        clusterStack.nextFreePos = 0;
        clusterActiveCount = (int)active_blocks;
        atomicExch(&flag, 0);
    }

    __syncthreads();
    cluster.sync();  // ensure all DSM state is initialized before traversal

    dualTraversalBlock<2, numConsumersPerBlock>(childOffsets, a, b,
                                               remoteStack, remoteFlag, authActiveCount,
                                               active_block,
                                               std::forward<MAC>(continuation),
                                               std::forward<M2L>(m2l),
                                               std::forward<P2P>(p2p));
}

template<int numConsumersPerBlock, class MAC, class M2L, class P2P>
__device__ void dualTraversalGPU(const TreeNodeIndex* __restrict__ childOffsets,
                                 TreeNodeIndex rootA, TreeNodeIndex rootB,
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

    if (!active_cluster) {
        if (block_rank_in_cluster() == 0 && thread_rank_in_block() == 0) printf("WARNING: TBC %u of %u is not active \n", cluster_id, numClusters);
        return;
    }

    dualTraversalTBC<numConsumersPerBlock>(childOffsets, a, b,
                                      std::forward<MAC>(continuation),
                                      std::forward<M2L>(m2l),
                                      std::forward<P2P>(p2p));
}

} // namespace cstone