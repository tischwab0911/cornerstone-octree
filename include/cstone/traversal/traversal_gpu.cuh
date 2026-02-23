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

#include "cstone/tree/octree.hpp"
#include "cstone/cuda/gpu_config.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#pragma nv_diag_suppress static_var_with_dynamic_init

HOST_DEVICE_FUN __forceinline__
constexpr std::size_t align_up(std::size_t x, std::size_t a)
{
    // a must be a power of two for this form
    return (x + (a - 1)) & ~(a - 1);
}



namespace cg = cooperative_groups;


namespace cstone
{

using NodePair = util::array<TreeNodeIndex, 2>;
struct WorkItem {
    NodePair nodePair = NodePair{0u,0u};
    bool valid = false;
    bool quit = false;
};

template<int Stages, int consumerMultiple, class MAC, class M2L, class P2P>
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
    __shared__ WorkItem bufferm2l[Stages][consumerMultiple * GpuConfig::warpSize];
    __shared__ WorkItem bufferp2p[Stages][consumerMultiple * GpuConfig::warpSize];
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
        WorkItem m2lItemBuffer[consumerMultiple];
        WorkItem p2pItemBuffer[consumerMultiple];

        bool stackContainsValues = true;
        while (stackContainsValues) {
            int stackPos = nextFreePos - (int)GpuConfig::warpSize + (int)blockThreadIdx;
            bool validPos = (stackPos >= 0 && stackPos < stackSize) ? 1 : 0;

            TreeNodeIndex nodeA = validPos ? nodeAStack[stackPos] : 0;
            TreeNodeIndex nodeB = validPos ? nodeBStack[stackPos] : 0;

            TreeNodeIndex nodeAChildOffset = validPos ? childOffsets[nodeA] : 0;
            TreeNodeIndex nodeBChildOffset = validPos ? childOffsets[nodeB] : 0;
            bool nodeAIsLeaf = nodeAChildOffset == 0;
            bool nodeBIsLeaf = nodeBChildOffset == 0;
            TreeNodeIndex divideNodeA = ((nodeA < nodeB && !nodeAIsLeaf) || nodeBIsLeaf) ? 1 : 0;

            int producedPairs = 0;

            #pragma unroll
            for (TreeNodeIndex octant = 0; octant < maxNewChildren; ++octant) {
                TreeNodeIndex nodeAChildIdx = validPos ? (1 - divideNodeA) * nodeA + (nodeAChildOffset + octant) * divideNodeA : 0u;
                TreeNodeIndex nodeBChildIdx = validPos ? divideNodeA * nodeB       + (nodeBChildOffset + octant) * (1 - divideNodeA) : 0u;
                bool continueTraversal = validPos ? continuation(nodeAChildIdx,nodeBChildIdx) : false;
                bool nodeAChildIsLeaf = childOffsets[nodeAChildIdx] == 0;
                bool nodeBChildIsLeaf = childOffsets[nodeBChildIdx] == 0;

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
                consumerIdx = (consumerIdx + 1)%consumerMultiple;

                if ((consumerIdx == 0) || (octant == maxNewChildren-1)) {
                    // put m2l items into the pipeline
                    m2lPipeline.producer_acquire();
                    #pragma unroll
                    for (int consumer = 0; consumer < consumerMultiple; ++consumer) {
                        if ((consumer + consumerMultiple * (octant/consumerMultiple)) >= maxNewChildren) {
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
                    for (int consumer = 0; consumer < consumerMultiple; ++consumer) {
                        if ((consumer + consumerMultiple * (octant/consumerMultiple)) >= maxNewChildren) {
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
                    for (int consumer = 0; consumer < consumerMultiple; ++consumer) {
                        cuda::memcpy_async(&bufferm2l[s][consumer * GpuConfig::warpSize + blockThreadIdx], &terminate,
                                            sizeof(WorkItem), m2lPipeline);
                    }
                    m2lPipeline.producer_commit();

                    p2pPipeline.producer_acquire();
                    #pragma unroll
                    for (int consumer = 0; consumer < consumerMultiple; ++consumer) {
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

template <int consumerMultiple, class MAC, class M2L, class P2P>
__device__ void dualTraversalTBC( const TreeNodeIndex* __restrict__ childOffsets,
                                        TreeNodeIndex a, TreeNodeIndex b,
                                        MAC&& continuation, M2L&& m2l, P2P&& p2p) {

    // __shared__ TreeNodeIndex nodeAClusterStack[128];
    // __shared__ TreeNodeIndex nodeBClusterStack[128];
    // __shared__ int nextFreePos;

    /* TODO: Distribute Work to blocks and set up TBC stack*/


    assert(blockDim.x%GpuConfig::warpSize == 0 && "blockDim must be multiple of GPU warpSize");

    dualTraversalBlock<2, consumerMultiple>(childOffsets, a, b,
                            std::forward<MAC>(continuation),
                            std::forward<M2L>(m2l),
                            std::forward<P2P>(p2p));
    
}

template<int consumerMultiple, class MAC, class M2L, class P2P>
__device__ void dualTraversalGPU(const TreeNodeIndex* __restrict__ childOffsets,
                                  TreeNodeIndex rootA, TreeNodeIndex rootB,
                                  MAC&& continuation, M2L&& m2l, P2P&& p2p) {

    const unsigned workIdx = blockIdx.x;
    const unsigned numClusters = gridDim.x;
    assert(numClusters == 8);
    TreeNodeIndex a = childOffsets[rootA] + workIdx;
    TreeNodeIndex b = rootB;

    /* TODO: Distribute work to TBCs*/
    /* TODO (long-term): investigate load balancing strategies (potentially introduce global stack)*/

    dualTraversalTBC<consumerMultiple>(childOffsets, a, b, std::forward<MAC>(continuation), 
                                           std::forward<M2L>(m2l), std::forward<P2P>(p2p));
}

} // namespace cstone