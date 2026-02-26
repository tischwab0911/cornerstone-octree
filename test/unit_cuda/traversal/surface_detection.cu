/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Cornerstone octree GPU testing
 *
 * @author Timo Schwab <tischwab@ethz.ch>
 *
 */

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>


#include "gtest/gtest.h"
#include "cstone/tree/octree_gpu.h"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/cs_util.hpp"
#include "cstone/traversal/macs.hpp"
#include "cstone/traversal/traversal.hpp"
#include "cstone/traversal/traversal_gpu.cuh"

namespace cstone {

template <int numConsumersPerBlock, class KeyType>
__global__ void dualTraversalNeighborsCount(
    const TreeNodeIndex* __restrict__ childOffsets,
    const KeyType* __restrict__ codeStarts,
    const KeyType* __restrict__ codeEnds,
    const unsigned* __restrict__ nodeLevels,
    KeyType focusStart,
    KeyType focusEnd,
    Box<float> box,
    TreeNodeIndex rootA,
    TreeNodeIndex rootB,
    util::array<TreeNodeIndex, 2>* p2pPairs,
    unsigned* p2pPairCount)
{
    auto crossFocusSurfacePairs =
        [focusStart, focusEnd, codeStarts, codeEnds, nodeLevels, box]
        __device__(TreeNodeIndex a, TreeNodeIndex b)
    {
        bool aFocusOverlap = overlapTwoRanges(focusStart, focusEnd, codeStarts[a], codeEnds[a]);
        bool bInFocus      = containedIn(codeStarts[b], codeEnds[b], focusStart, focusEnd);
        if (!aFocusOverlap || bInFocus) { return false; }

        IBox aBox = sfcIBox(sfcKey(codeStarts[a]), nodeLevels[a]);
        IBox bBox = sfcIBox(sfcKey(codeStarts[b]), nodeLevels[b]);
        return minDistanceSq<KeyType>(aBox, bBox, box) == 0.0;
    };

    auto m2l = [] __device__(TreeNodeIndex, TreeNodeIndex) {};

    auto p2p = [p2pPairs, p2pPairCount] __device__(TreeNodeIndex a, TreeNodeIndex b) {
        unsigned idx = atomicAdd(p2pPairCount, 1u);
        p2pPairs[idx][0] = a;
        p2pPairs[idx][1] = b;
    };

    dualTraversalGPU<numConsumersPerBlock>(childOffsets, rootA, rootB,
                                           crossFocusSurfacePairs, m2l, p2p);
}

struct TravConfig {

    //! @brief number of consumer warps ber block, all warps except warp 0 are consumers
    static constexpr unsigned numConsumersPerBlock = 3;

    /*! @brief number of threads per block for the traversal kernel
     * number of threads per block for the dual traversal kernel
     * must be at least 64 and at most 512
     * must be a multiple of GPU warp size
     */
    static constexpr unsigned numThreadsPerBlock = (numConsumersPerBlock + 1) * GpuConfig::warpSize;
    static_assert(numThreadsPerBlock >= 64 && numThreadsPerBlock <= 512);

    //! @brief number of blocks per thread block cluster, should be a power of 8: (1, 8, 64, ...)
    static constexpr unsigned kBlocksPerCluster = 8;

    //! @brief number of TBS in grid, should be a power of 8: (1, 8, 64, ...)
    static constexpr unsigned kNumClusters      = 8;

    //! @brief total number of blocks launched in the grid
    static constexpr unsigned kTotalBlocks      = kBlocksPerCluster * kNumClusters;
    static_assert(kBlocksPerCluster > 0 && kNumClusters > 0);

    static constexpr unsigned ClusterStackSize = GpuConfig::warpSize * kBlocksPerCluster;
    static constexpr unsigned OverflowLevel = GpuConfig::warpSize;

};


template <class KeyType>
void dualTraversalNeighborsGpu()
{
    // ── Build CPU tree and compute reference ─────────────────────
    Octree<KeyType> octree;
    auto leaves = makeUniformNLevelTree<KeyType>(64, 1);
    octree.update(leaves.data(), nNodes(leaves));

    Box<float> box(0, 1);

    KeyType focusStart = octree.codeStart(octree.toInternal(0));
    KeyType focusEnd   = octree.codeStart(octree.toInternal(8));

    // CPU reference pairs
    std::vector<util::array<TreeNodeIndex, 2>> cpuPairs;
    auto crossFocusSurfacePairs = [focusStart, focusEnd, &tree = octree, &box](
                                      TreeNodeIndex a, TreeNodeIndex b)
    {
        bool aFocusOverlap = overlapTwoRanges(focusStart, focusEnd, tree.codeStart(a), tree.codeEnd(a));
        bool bInFocus      = containedIn(tree.codeStart(b), tree.codeEnd(b), focusStart, focusEnd);
        if (!aFocusOverlap || bInFocus) { return false; }

        IBox aBox = sfcIBox(sfcKey(tree.codeStart(a)), tree.level(a));
        IBox bBox = sfcIBox(sfcKey(tree.codeStart(b)), tree.level(b));
        return minDistanceSq<KeyType>(aBox, bBox, box) == 0.0;
    };
    auto m2lCpu = [](TreeNodeIndex, TreeNodeIndex) {};
    auto p2pCpu = [&cpuPairs](TreeNodeIndex a, TreeNodeIndex b) { cpuPairs.push_back({a, b}); };

    dualTraversal(octree.childOffsets().data(), 0, 0, crossFocusSurfacePairs, m2lCpu, p2pCpu);
    std::sort(cpuPairs.begin(), cpuPairs.end());
    EXPECT_EQ(cpuPairs.size(), 61);

    // ── Extract per-node arrays from CPU tree ────────────────────
    const TreeNodeIndex numNodes = octree.numTreeNodes();

    std::vector<KeyType>  h_codeStarts(numNodes);
    std::vector<KeyType>  h_codeEnds(numNodes);
    std::vector<unsigned> h_levels(numNodes);

    for (TreeNodeIndex i = 0; i < numNodes; ++i)
    {
        h_codeStarts[i] = octree.codeStart(i);
        h_codeEnds[i]   = octree.codeEnd(i);
        h_levels[i]     = octree.level(i);
    }

    // ── Upload tree data to GPU ──────────────────────────────────
    DeviceVector<KeyType>  d_leaves = leaves;
    OctreeData<KeyType, GpuTag> gpuTree;
    gpuTree.resize(nNodes(leaves));
    buildOctreeGpu(rawPtr(d_leaves), gpuTree.data());

    DeviceVector<KeyType>  d_codeStarts = h_codeStarts;
    DeviceVector<KeyType>  d_codeEnds   = h_codeEnds;
    DeviceVector<unsigned> d_levels     = h_levels;

    // ── Allocate output buffers ──────────────────────────────────
    const unsigned maxPairs = 256;
    DeviceVector<util::array<TreeNodeIndex, 2>> d_p2pPairs(maxPairs);

    unsigned* d_p2pCount;
    cudaMalloc(&d_p2pCount, sizeof(unsigned));
    cudaMemset(d_p2pCount, 0, sizeof(unsigned));

    // ── Launch configuration ─────────────────────────────────────
    dim3 block(TravConfig::numThreadsPerBlock, 1, 1);
    dim3 grid(TravConfig::kTotalBlocks, 1, 1);

    cudaLaunchConfig_t cfg{};
    cfg.gridDim  = grid;
    cfg.blockDim = block;

    cudaLaunchAttribute attr{};
    attr.id                = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x  = TravConfig::kBlocksPerCluster;
    attr.val.clusterDim.y  = 1;
    attr.val.clusterDim.z  = 1;

    cfg.attrs    = &attr;
    cfg.numAttrs = 1;

    cudaLaunchKernelEx(&cfg,
                       dualTraversalNeighborsCount<TravConfig::numConsumersPerBlock, KeyType>,
                       rawPtr(gpuTree.childOffsets),
                       rawPtr(d_codeStarts),
                       rawPtr(d_codeEnds),
                       rawPtr(d_levels),
                       focusStart,
                       focusEnd,
                       box,
                       0, 0,
                       rawPtr(d_p2pPairs),
                       d_p2pCount);

    cudaDeviceSynchronize();

    // ── Copy back and validate ───────────────────────────────────
    unsigned h_p2pCount = 0;
    cudaMemcpy(&h_p2pCount, d_p2pCount, sizeof(unsigned), cudaMemcpyDeviceToHost);

    std::vector<util::array<TreeNodeIndex, 2>> h_p2pPairs(h_p2pCount);
    cudaMemcpy(h_p2pPairs.data(), rawPtr(d_p2pPairs),
               h_p2pCount * sizeof(util::array<TreeNodeIndex, 2>),
               cudaMemcpyDeviceToHost);

    std::sort(h_p2pPairs.begin(), h_p2pPairs.end());
    h_p2pPairs.erase(std::unique(h_p2pPairs.begin(), h_p2pPairs.end()), h_p2pPairs.end());

    printf("Expected P2P Pairs: %zu, GPU Computed P2P Pairs: %zu\n",
           cpuPairs.size(), h_p2pPairs.size());

    EXPECT_EQ(h_p2pPairs.size(), cpuPairs.size());
    for (size_t i = 0; i < cpuPairs.size(); ++i)
    {
        EXPECT_EQ(h_p2pPairs[i], cpuPairs[i]);
    }

    // ── Validate pair properties (same checks as CPU test) ───────
    for (auto p : h_p2pPairs)
    {
        auto a = p[0];
        auto b = p[1];
        // a in focus
        EXPECT_TRUE(h_codeStarts[a] >= focusStart && h_codeEnds[a] <= focusEnd);
        // b outside focus
        EXPECT_TRUE(h_codeStarts[b] >= focusEnd || h_codeEnds[a] <= focusStart);
        // a and b touch each other
        IBox aBox = sfcIBox(sfcKey(h_codeStarts[a]), h_levels[a]);
        IBox bBox = sfcIBox(sfcKey(h_codeStarts[b]), h_levels[b]);
        EXPECT_FLOAT_EQ((minDistanceSq<KeyType>(aBox, bBox, box)), 0.0);
    }

    cudaFree(d_p2pCount);
}

TEST(Traversal, dualTraversalNeighborsGpu)
{
    dualTraversalNeighborsGpu<unsigned>();
    dualTraversalNeighborsGpu<uint64_t>();
}

} // namespace cstone