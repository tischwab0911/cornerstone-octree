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
    NodePair nodePair = {0,0};
    bool valid = false;
    bool quit = false;
};

template<int Stages, class MAC, class M2L, class P2P>
__device__ void dualTraversalBlock( const TreeNodeIndex* __restrict__ childOffsets,
                                    TreeNodeIndex a, TreeNodeIndex b,
                                    MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    using State = cuda::pipeline_shared_state<cuda::thread_scope_block, Stages>;
    constexpr TreeNodeIndex maxNewChildren = 8;
    constexpr cuda::std::size_t numWarpSpecializations = 2;

    // get thread block and determine size
    cg::thread_block block = cg::this_thread_block();
    cuda::std::size_t blockSize = block.size();
    constexpr cuda::std::size_t maxTileSize = GpuConfig::warpSize;
    assert(blockSize == 2*GpuConfig::warpSize && "only launch blocks with 64 threads");

    // half the threads are producers, half are consumers
    cuda::std::size_t numProdConsLanes = blockSize / numWarpSpecializations;
    // auto tile = cg::tiled_partition<numProdConsLanes>(block);

    // determine if thread is producer or consumer
    // tileId == 0: producer, tileId == 1: consumer
    cuda::std::size_t blockThreadIdx = block.thread_rank();
    cuda::std::size_t tileThreadIdx  = blockThreadIdx % numProdConsLanes;
    cuda::std::size_t tileId = blockThreadIdx / numProdConsLanes;
    cuda::std::size_t lastLaneInTile = numProdConsLanes - 1;

    // if(threadIdx.x == 0) printf("%u\n", blockIdx.x);

    // we may have to switch to SoA somehow in the future :/
    // for now this is easier, I will take a look at the nsight-compute report
    // to see how costly the bank conflicts are
    __shared__ WorkItem bufferm2l[Stages][GpuConfig::warpSize];
    __shared__ WorkItem bufferp2p[Stages][GpuConfig::warpSize];

    // WorkItem (*bufferm2l)[maxTileSize] = reinterpret_cast<WorkItem (*)[maxTileSize]>(shared_mem);
    // WorkItem (*bufferp2p)[maxTileSize] = reinterpret_cast<WorkItem (*)[maxTileSize]>(shared_mem + sizeof(WorkItem) * Stages * maxTileSize);

    // // align pointers in shmem for pipelines
    // size_t offset = 2 * sizeof(WorkItem) * Stages * maxTileSize;
    // offset = align_up(offset, alignof(State));
    // auto *ptrm2l = reinterpret_cast<State*>(shared_mem + offset);
    // offset = align_up(offset + sizeof(State), alignof(State));
    // auto *ptrp2p = reinterpret_cast<State*>(shared_mem + offset);
    __shared__ State ptrm2l;
    __shared__ State ptrp2p;

    auto thread_role = (tileId == 0) ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;
    auto m2lPipeline = cuda::make_pipeline(block, &ptrm2l, thread_role);
    auto p2pPipeline = cuda::make_pipeline(block, &ptrp2p, thread_role);

    if (tileId == 0) {

        auto tile = cg::coalesced_threads();

        int prodStageIdx = 0;
        constexpr int stackSize = maxTileSize * maxNewChildren * 2;

        __shared__ __align__(alignof(TreeNodeIndex)) TreeNodeIndex nodeAStack[stackSize];
        __shared__ __align__(alignof(TreeNodeIndex)) TreeNodeIndex nodeBStack[stackSize];

        TreeNodeIndex nodeABuffer[8];
        TreeNodeIndex nodeBBuffer[8];

        int nextFreePos = 1;

        if (tileThreadIdx == 0) {
            nodeAStack[0] = a;
            nodeBStack[0] = b;
        }

        bool stackContainsValues = true;

        while (stackContainsValues) {
            int stackPos = nextFreePos - (int)numProdConsLanes + (int)tileThreadIdx;
            bool validPos = (stackPos >= 0 && stackPos < stackSize) ? 1 : 0;

            TreeNodeIndex nodeA = validPos ? nodeAStack[stackPos] : 0;
            TreeNodeIndex nodeB = validPos ? nodeBStack[stackPos] : 0;

            TreeNodeIndex nodeAChildOffset = validPos ? childOffsets[nodeA] : 0;
            TreeNodeIndex nodeBChildOffset = validPos ? childOffsets[nodeB] : 0;
            bool nodeAIsLeaf = nodeAChildOffset == 0;
            bool nodeBIsLeaf = nodeBChildOffset == 0;
            // bool bothLeaves = nodeAIsLeaf && nodeBIsLeaf;
            TreeNodeIndex divideNodeA = ((nodeA < nodeB && !nodeAIsLeaf) || nodeBIsLeaf) ? 1 : 0;

            int producedPairs = 0;

            #pragma unroll
            for (TreeNodeIndex octant = 0; octant < maxNewChildren; ++octant) {
                TreeNodeIndex nodeAChildIdx = validPos ? (1 - divideNodeA) * nodeA + (nodeAChildOffset + octant) * divideNodeA : 0u;
                TreeNodeIndex nodeBChildIdx = validPos ? divideNodeA * nodeB       + (nodeBChildOffset + octant) * (1 - divideNodeA) : 0u;
                bool continueTraversal = validPos ? continuation(nodeAChildIdx,nodeBChildIdx) : false;
                bool nodeAChildIsLeaf = childOffsets[nodeAChildIdx] == 0;
                bool nodeBChildIsLeaf = childOffsets[nodeBChildIdx] == 0;

                // case continuation == 1 && bothleaves == 0, adds NodePair to local buffer
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

                m2lPipeline.producer_acquire();
                cuda::memcpy_async(&bufferm2l[prodStageIdx][tileThreadIdx], &m2lItem,
                                    sizeof(WorkItem), m2lPipeline);
                m2lPipeline.producer_commit();
                
                p2pPipeline.producer_acquire();
                cuda::memcpy_async(&bufferp2p[prodStageIdx][tileThreadIdx], &p2pItem,
                                    sizeof(WorkItem), p2pPipeline);
                p2pPipeline.producer_commit();

                prodStageIdx = (prodStageIdx + 1) % Stages;
            }
            
            int elementPopped = validPos ? 1 : 0;
            int numPopped = cg::inclusive_scan(tile, elementPopped, cg::plus<int>());
            int numToPush = cg::exclusive_scan(tile, producedPairs, cg::plus<int>());
            tile.sync();

            int baseWritePos = 0;

            if (tileThreadIdx == lastLaneInTile) {
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
            terminate.quit = true;
            if (!stackContainsValues) {
                for(int s = 0; s < Stages; ++s) {
                    m2lPipeline.producer_acquire();
                    cuda::memcpy_async(&bufferm2l[s][tileThreadIdx], &terminate,
                                        sizeof(WorkItem), m2lPipeline);
                    m2lPipeline.producer_commit();

                    p2pPipeline.producer_acquire();
                    cuda::memcpy_async(&bufferp2p[s][tileThreadIdx], &terminate,
                                        sizeof(WorkItem), p2pPipeline);
                    p2pPipeline.producer_commit();
                }
            }
        }

        m2lPipeline.quit();
        p2pPipeline.quit();

    } else if (tileId == 1) {

        int consStageIdx = 0;
        int release_signal = 0;

        while (release_signal < Stages) {
            m2lPipeline.consumer_wait();
            WorkItem m2lItem = bufferm2l[consStageIdx][tileThreadIdx];
            if (m2lItem.valid) {
                m2l(m2lItem.nodePair[0],m2lItem.nodePair[1]);
                // atomicAdd(&consumedm2l, 1u);
            } else if (m2lItem.quit) {
                ++release_signal;
            }
            m2lPipeline.consumer_release();

            p2pPipeline.consumer_wait();
            WorkItem p2pItem = bufferp2p[consStageIdx][tileThreadIdx];
            if (p2pItem.valid) {
                p2p(p2pItem.nodePair[0],p2pItem.nodePair[1]);
                // int number = atomicAdd(&consumedp2p, 1u);
                // printf("iteration: %d, nodePair: (%u,%u)\n", number, p2pItem.nodePair[0],p2pItem.nodePair[1]);
            }
            p2pPipeline.consumer_release();
            consStageIdx = (consStageIdx + 1) % Stages;
        }
        m2lPipeline.quit();
        p2pPipeline.quit();
    } 
}

template <class MAC, class M2L, class P2P>
__device__ void dualTraversalCluster( const TreeNodeIndex* __restrict__ childOffsets,
                                        TreeNodeIndex a, TreeNodeIndex b,
                                        MAC&& continuation, M2L&& m2l, P2P&& p2p) {

    // __shared__ TreeNodeIndex nodeAClusterStack[128];
    // __shared__ TreeNodeIndex nodeBClusterStack[128];
    // __shared__ int nextFreePos;

    dualTraversalBlock<2>(childOffsets, a, b,
                            std::forward<MAC>(continuation),
                            std::forward<M2L>(m2l),
                            std::forward<P2P>(p2p));
    
}

} // namespace cstone