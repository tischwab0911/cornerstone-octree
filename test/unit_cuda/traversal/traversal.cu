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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <iostream>


#include "gtest/gtest.h"
#include "cstone/tree/octree_gpu.h"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/cs_util.hpp"
#include "cstone/traversal/macs.hpp"
#include "cstone/traversal/traversal.hpp"
#include "cstone/traversal/traversal_gpu.cuh"

namespace cstone
{

template<int consumerMultiple>
__global__ void dualTraversalGrid(const TreeNodeIndex* __restrict__ childOffsets,
                                        TreeNodeIndex rootA,
                                        TreeNodeIndex rootB,
                                        util::array<TreeNodeIndex, 2>* pairs,
                                        unsigned* pairCount)
{
    const unsigned workIdx = blockIdx.x;
    const unsigned numClusters = gridDim.x;
    assert(numClusters == 8);
    rootA = childOffsets[rootA] + workIdx;

    // admissibility criterion: accept all internal node pairs
    auto allPairs = [] __device__(TreeNodeIndex, TreeNodeIndex) { return true; };
    // multipole‑to‑local interaction (unused here)
    auto m2l = [] __device__(TreeNodeIndex, TreeNodeIndex) {};
    // particle‑to‑particle interaction: record each leaf pair atomically
    auto p2p = [pairs, pairCount] __device__(TreeNodeIndex a, TreeNodeIndex b) {
        unsigned idx = atomicAdd(pairCount, 1u);
        pairs[idx][0] = a;
        pairs[idx][1] = b;
    };
    // Perform the dual traversal starting from the roots.  The pointer
    // childOffsets refers to the GPU octree’s child pointer array.
    dualTraversalCluster<consumerMultiple>(childOffsets, rootA, rootB,
                         allPairs, m2l, p2p);
}

template <class KeyType>
void dualTraversalAllPairsGpu()
{
    constexpr int numThreadsPerBlock = 9*32;
    constexpr int consumerMultiple = (numThreadsPerBlock - GpuConfig::warpSize) / GpuConfig::warpSize;
    // Build a simple tree on the CPU with 22 leaves.  The tree
    // structure follows the same construction as in the CPU test:
    // start with one node, split once, then split child 0 three more
    // times to get a total of 22 leaf nodes.
    Octree<KeyType> cpuTree;
    auto leaves = OctreeMaker<KeyType>{}.divide().divide(0).divide(0, 7).divide(7).makeTree();
    cpuTree.update(leaves.data(), nNodes(leaves));

    // Compute the reference set of leaf‑pairs on the CPU using the
    // existing dualTraversal implementation.
    std::vector<util::array<TreeNodeIndex, 2>> cpuPairs;
    auto allPairsCpu = [](TreeNodeIndex, TreeNodeIndex) { return true; };
    auto m2lCpu = [](TreeNodeIndex, TreeNodeIndex) {};
    auto p2pCpu = [&cpuPairs](TreeNodeIndex a, TreeNodeIndex b) {
        cpuPairs.push_back({a, b});
    };
    dualTraversal(cpuTree.childOffsets().data(), 0, 0, allPairsCpu, m2lCpu, p2pCpu);
    std::sort(cpuPairs.begin(), cpuPairs.end());
    cpuPairs.erase(std::unique(cpuPairs.begin(), cpuPairs.end()), cpuPairs.end());

    // Build the GPU representation of the same tree.
    DeviceVector<KeyType> d_leaves = leaves;
    OctreeData<KeyType, GpuTag> gpuTree;
    gpuTree.resize(nNodes(leaves));
    buildOctreeGpu(rawPtr(d_leaves), gpuTree.data());

    // Allocate a device buffer large enough to hold all possible pairs.
    const unsigned maxPairs = static_cast<unsigned>(cpuPairs.size());
    DeviceVector<util::array<TreeNodeIndex, 2>> d_pairs(maxPairs);
    unsigned* d_count;
    cudaMalloc(&d_count, sizeof(unsigned));
    cudaMemset(d_count, 0, sizeof(unsigned));

    dim3 block(numThreadsPerBlock,1,1);
    dim3 grid(8,1,1);
    dualTraversalGrid<consumerMultiple><<<grid, block>>>(
        rawPtr(gpuTree.childOffsets), 0, 0, rawPtr(d_pairs), d_count);

    cudaDeviceSynchronize();

    // Copy the count and pairs back to the host.
    unsigned h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(unsigned), cudaMemcpyDeviceToHost);
    std::vector<util::array<TreeNodeIndex, 2>> h_pairs;
    h_pairs.resize(h_count);
    cudaMemcpy(h_pairs.data(), rawPtr(d_pairs), h_count * sizeof(util::array<TreeNodeIndex, 2>),
               cudaMemcpyDeviceToHost);
    std::sort(h_pairs.begin(), h_pairs.end());
    h_pairs.erase(std::unique(h_pairs.begin(), h_pairs.end()), h_pairs.end());

    // Compare GPU results against CPU reference.
    EXPECT_EQ(h_pairs.size(), cpuPairs.size());
    for (size_t i = 0; i < cpuPairs.size(); ++i)
    {
        EXPECT_EQ(h_pairs[i], cpuPairs[i]);
    }

    cudaFree(d_count);
}

TEST(Traversal, dualTraversalAllPairsGpu)
{
    // Run the GPU dual traversal for both 32‑bit and 64‑bit octree keys.
    dualTraversalAllPairsGpu<unsigned>();
    dualTraversalAllPairsGpu<uint64_t>();
}

} // namespace cstone