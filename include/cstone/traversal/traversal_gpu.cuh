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
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace cg = cooperative_groups;

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

HOST_DEVICE_FUN __forceinline__
constexpr std::size_t align_up(std::size_t x, std::size_t a)
{
    // a must be a power of two for this form
    return (x + (a - 1)) & ~(a - 1);
}


namespace cstone
{

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
    unsigned L = 0, cap = 1;
    while (cap < n) { cap *= 8; ++L; }
    return L;
}

__device__ __forceinline__ unsigned pow8(unsigned L)
{
    unsigned v = 1;
    while (L--) v *= 8;
    return v;
}

__device__ __forceinline__ void decode_base8_digits(unsigned rid, unsigned L, unsigned* digits)
{
    // digits[0] MS digit, digits[L-1] LS digit
    for (unsigned i = 0; i < L; ++i) digits[i] = 0;
    for (unsigned i = 0; i < L; ++i) {
        digits[L - 1 - i] = rid % 8;
        rid /= 8;
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
    unsigned digits_safe[16];          // prefix digits that were safely applied

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

        if (a < b) a = childOffsets[a] + oct;
        else       b = childOffsets[b] + oct;

        // commit this level as safe
        digits_safe[level] = oct;
        a_safe = a; b_safe = b;
        L_safe = level + 1;
    }

    const unsigned fanout = pow8(L_safe);
    active_count = (fanout < N) ? fanout : N;

    // If we couldn’t safely split even once, fanout=1 -> only rid==0 active.
    if (rid >= active_count) return false;
    return true;
}

using NodePair = util::array<TreeNodeIndex, 2>;
struct WorkItem {
    NodePair nodePair = NodePair{0u,0u};
    bool valid = false;
    bool quit = false;
};

template<int Stages, int numConsumersPerBlock, class MAC, class M2L, class P2P>
__device__ void dualTraversalBlock( const TreeNodeIndex* __restrict__ childOffsets,
                                    TreeNodeIndex a, TreeNodeIndex b,
                                    MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    using State = cuda::pipeline_shared_state<cuda::thread_scope_block, Stages>;
    constexpr TreeNodeIndex maxNewChildren = 8;

    // get thread block and determine size
    cg::thread_block block = cg::this_thread_block();
    cuda::std::size_t blockSize = block.size();
    cuda::std::size_t blockThreadIdx = block.thread_rank();

    // we may have to switch to SoA somehow in the future :/
    // for now this is easier, I will take a look at the nsight-compute report
    // to see how costly the bank conflicts are
    __shared__ WorkItem bufferm2l[Stages][numConsumersPerBlock * GpuConfig::warpSize];
    __shared__ WorkItem bufferp2p[Stages][numConsumersPerBlock * GpuConfig::warpSize];
    __shared__ State ptrm2l;
    __shared__ State ptrp2p;

    auto thread_role = (blockThreadIdx < GpuConfig::warpSize) ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;
    auto m2lPipeline = cuda::make_pipeline(block, &ptrm2l, thread_role);
    auto p2pPipeline = cuda::make_pipeline(block, &ptrp2p, thread_role);

    if (blockThreadIdx < GpuConfig::warpSize) {

        auto tile = cg::coalesced_threads();
        int lastLaneInTile = tile.size()-1;

        int prodStageIdx = 0; 
        int consumerIdx = 0;

        constexpr int stackSize = GpuConfig::warpSize * maxNewChildren * 2;
        __shared__ __align__(alignof(TreeNodeIndex)) TreeNodeIndex nodeAStack[stackSize];
        __shared__ __align__(alignof(TreeNodeIndex)) TreeNodeIndex nodeBStack[stackSize];

        // these are the buffers for the nodepairs that are being pushed
        // back onto the traversal stack. They are necessary because pairs
        // are not pushed before traversing all 8 potential child octants
        TreeNodeIndex nodeABuffer[maxNewChildren];
        TreeNodeIndex nodeBBuffer[maxNewChildren];

        // dummmy work item
        WorkItem dummy;
        dummy.nodePair = NodePair{0,0};
        dummy.valid = false;
        dummy.quit = false;
        
        // write nodepair from function call into 0th position
        // and set next free position to 1
        if (blockThreadIdx == 0) {
            nodeAStack[0] = a;
            nodeBStack[0] = b;
        }
        int nextFreePos = 1;
        
        // these are necessary because when we k-times more consumers than
        // producers, we only push every k-th octant iteration to the octree
        WorkItem m2lItemBuffer[numConsumersPerBlock];
        WorkItem p2pItemBuffer[numConsumersPerBlock];

        bool stackContainsValues = true;
        while (stackContainsValues) {
            int stackPos = nextFreePos - (int)GpuConfig::warpSize + (int)blockThreadIdx;
            bool validPos = (stackPos >= 0 && stackPos < stackSize) ? 1 : 0;

            TreeNodeIndex nodeA = validPos ? nodeAStack[stackPos] : 0;
            TreeNodeIndex nodeB = validPos ? nodeBStack[stackPos] : 0;

            TreeNodeIndex nodeAChildOffset = validPos ? childOffsets[nodeA] : 0;
            TreeNodeIndex nodeBChildOffset = validPos ? childOffsets[nodeB] : 0;
            bool nodeAIsLeaf = isLeaf(childOffsets, nodeA);
            bool nodeBIsLeaf = isLeaf(childOffsets, nodeB);
            TreeNodeIndex divideNodeA = ((nodeA < nodeB && !nodeAIsLeaf) || nodeBIsLeaf) ? 1 : 0;

            int producedPairs = 0;

            #pragma unroll
            for (TreeNodeIndex octant = 0; octant < maxNewChildren; ++octant) {
                TreeNodeIndex nodeAChildIdx = validPos ? (1 - divideNodeA) * nodeA + (nodeAChildOffset + octant) * divideNodeA : 0u;
                TreeNodeIndex nodeBChildIdx = validPos ? divideNodeA * nodeB       + (nodeBChildOffset + octant) * (1 - divideNodeA) : 0u;
                bool continueTraversal = validPos ? continuation(nodeAChildIdx,nodeBChildIdx) : false;
                bool nodeAChildIsLeaf = isLeaf(childOffsets, nodeAChildIdx);
                bool nodeBChildIsLeaf = isLeaf(childOffsets, nodeBChildIdx);

                int addPairToLocalStack = (continueTraversal && !(nodeAChildIsLeaf && nodeBChildIsLeaf)) ? 1 : 0;
                if (addPairToLocalStack) {
                    nodeABuffer[producedPairs] = nodeAChildIdx;
                    nodeBBuffer[producedPairs] = nodeBChildIdx;
                    ++producedPairs;
                }

                NodePair node = NodePair{nodeAChildIdx,nodeBChildIdx};
                WorkItem m2lItem;
                WorkItem p2pItem;
            
                m2lItem.nodePair = node;
                p2pItem.nodePair = node;
                m2lItem.valid = (!addPairToLocalStack && !continueTraversal && (validPos == 1));
                p2pItem.valid = (!addPairToLocalStack && continueTraversal);

                p2pItemBuffer[consumerIdx] = p2pItem;
                m2lItemBuffer[consumerIdx] = m2lItem;
                consumerIdx = (consumerIdx + 1)%numConsumersPerBlock;

                if ((consumerIdx == 0) || (octant == maxNewChildren-1)) {
                    // put m2l items into the pipeline
                    m2lPipeline.producer_acquire();
                    #pragma unroll
                    for (int consumer = 0; consumer < numConsumersPerBlock; ++consumer) {
                        if ((consumer + numConsumersPerBlock * (octant/numConsumersPerBlock)) >= maxNewChildren) {
                            m2lItemBuffer[consumer]=dummy;
                            consumerIdx = 0;
                        }
                        cuda::memcpy_async(&bufferm2l[prodStageIdx][consumer * GpuConfig::warpSize + blockThreadIdx], &m2lItemBuffer[consumer],
                                            sizeof(WorkItem), m2lPipeline);
                    }
                    m2lPipeline.producer_commit();

                    // put p2p items into the pipeline
                    p2pPipeline.producer_acquire();
                    #pragma unroll
                    for (int consumer = 0; consumer < numConsumersPerBlock; ++consumer) {
                        if ((consumer + numConsumersPerBlock * (octant/numConsumersPerBlock)) >= maxNewChildren) {
                            p2pItemBuffer[consumer] = dummy;
                            consumerIdx = 0;
                        }
                        cuda::memcpy_async(&bufferp2p[prodStageIdx][consumer * GpuConfig::warpSize + blockThreadIdx], &p2pItemBuffer[consumer],
                                            sizeof(WorkItem), p2pPipeline);
                    }
                    p2pPipeline.producer_commit();

                    prodStageIdx = (prodStageIdx + 1)%Stages;
                }
            }
            
            int elementPopped = validPos ? 1 : 0;
            int numPopped = cg::inclusive_scan(tile, elementPopped, cg::plus<int>());
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

            if (nextFreePos == 0) stackContainsValues = false;

            // add pairs from buffer to shmem TODO
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i < producedPairs) {
                    if((writeNewPos + i) >= stackSize) assert(false);
                    nodeAStack[writeNewPos + i] = nodeABuffer[i];
                    nodeBStack[writeNewPos + i] = nodeBBuffer[i];
                }
            }

            WorkItem terminate;
            terminate.valid = false;
            terminate.quit = true;
            if (!stackContainsValues) {
                for(int stage = 0; stage < Stages; ++stage) {
                    int s = (prodStageIdx+stage)%Stages;
                    m2lPipeline.producer_acquire();
                    #pragma unroll
                    for (int consumer = 0; consumer < numConsumersPerBlock; ++consumer) {
                        cuda::memcpy_async(&bufferm2l[s][consumer * GpuConfig::warpSize + blockThreadIdx], &terminate,
                                            sizeof(WorkItem), m2lPipeline);
                    }
                    m2lPipeline.producer_commit();

                    p2pPipeline.producer_acquire();
                    #pragma unroll
                    for (int consumer = 0; consumer < numConsumersPerBlock; ++consumer) {
                        cuda::memcpy_async(&bufferp2p[s][consumer * GpuConfig::warpSize + blockThreadIdx], &terminate,
                                            sizeof(WorkItem), p2pPipeline);
                    }
                    p2pPipeline.producer_commit();
                }
            }
        }

        m2lPipeline.quit();
        p2pPipeline.quit();

    } else if (blockThreadIdx >= GpuConfig::warpSize) {

        int consStageIdx = 0;
        int release_signal = 0;
        int tileThreadIdx = blockThreadIdx - GpuConfig::warpSize;

        while (release_signal < Stages) {
            m2lPipeline.consumer_wait();
            WorkItem m2lItem = bufferm2l[consStageIdx][tileThreadIdx];
            if (m2lItem.valid) {
                m2l(m2lItem.nodePair[0],m2lItem.nodePair[1]);
            } else if (m2lItem.quit) {
                ++release_signal;
            }
            m2lPipeline.consumer_release();

            p2pPipeline.consumer_wait();
            WorkItem p2pItem = bufferp2p[consStageIdx][tileThreadIdx];
            if (p2pItem.valid) {
                p2p(p2pItem.nodePair[0],p2pItem.nodePair[1]);
            }
            p2pPipeline.consumer_release();
            consStageIdx = (consStageIdx + 1) % Stages;
        }
        m2lPipeline.quit();
        p2pPipeline.quit();
    } 
}

template <int numConsumersPerBlock, class MAC, class M2L, class P2P>
__device__ void dualTraversalTBC(const TreeNodeIndex* __restrict__ childOffsets,
                                 TreeNodeIndex a, TreeNodeIndex b,
                                 MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    const unsigned block_in_cluster = block_rank_in_cluster();
    const unsigned blocksPerCluster = blocks_per_cluster_runtime();

    assert(blockDim.x % GpuConfig::warpSize == 0);

    unsigned active_blocks = 0;
    const bool active_block =
        assignPairBySplitting_regress(childOffsets, a, b,
                                      block_in_cluster, blocksPerCluster,
                                      active_blocks,
                                      std::forward<MAC>(continuation));

    if (!active_block) return;

    dualTraversalBlock<2, numConsumersPerBlock>(childOffsets, a, b,
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

    if (!active_cluster) return;

    dualTraversalTBC<numConsumersPerBlock>(childOffsets, a, b,
                                      std::forward<MAC>(continuation),
                                      std::forward<M2L>(m2l),
                                      std::forward<P2P>(p2p));
}

} // namespace cstone