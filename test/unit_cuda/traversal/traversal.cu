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
__global__ void dualTraversalCount(const TreeNodeIndex* __restrict__ childOffsets,
                                        TreeNodeIndex rootA,
                                        TreeNodeIndex rootB,
                                        util::array<TreeNodeIndex, 2>* p2pPairs,
                                        util::array<TreeNodeIndex, 2>* m2lPairs,
                                        unsigned* p2pPairCount,
                                        unsigned* m2lPairCount)
{
    // admissibility criterion: accept all internal node pairs
    auto allPairs = [] __device__(TreeNodeIndex, TreeNodeIndex) { return true; };
    // multipole‑to‑local interaction (unused here)
    auto m2l = [m2lPairs, m2lPairCount] __device__(TreeNodeIndex a, TreeNodeIndex b) {
        unsigned idx = atomicAdd(m2lPairCount, 1u);
        m2lPairs[idx][0] = a;
        m2lPairs[idx][1] = b;
    };
    // particle‑to‑particle interaction: record each leaf pair atomically
    auto p2p = [p2pPairs, p2pPairCount] __device__(TreeNodeIndex a, TreeNodeIndex b) {
        unsigned idx = atomicAdd(p2pPairCount, 1u);
        p2pPairs[idx][0] = a;
        p2pPairs[idx][1] = b;
    };
    // Perform the dual traversal starting from the roots.  The pointer
    // childOffsets refers to the GPU octree’s child pointer array.
    dualTraversalGPU<consumerMultiple>(childOffsets, rootA, rootB,
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
    std::vector<util::array<TreeNodeIndex, 2>> cpum2lPairs;
    std::vector<util::array<TreeNodeIndex, 2>> cpup2pPairs;
    auto allPairsCpu = [](TreeNodeIndex, TreeNodeIndex) { return true; };
    auto m2lCpu = [&cpum2lPairs](TreeNodeIndex a, TreeNodeIndex b) {
        cpum2lPairs.push_back({a, b});
    };
    auto p2pCpu = [&cpup2pPairs](TreeNodeIndex a, TreeNodeIndex b) {
        cpup2pPairs.push_back({a, b});
    };
    dualTraversal(cpuTree.childOffsets().data(), 0, 0, allPairsCpu, m2lCpu, p2pCpu);
    std::sort(cpup2pPairs.begin(), cpup2pPairs.end());
    std::sort(cpum2lPairs.begin(), cpum2lPairs.end());
    
    cpup2pPairs.erase(std::unique(cpup2pPairs.begin(), cpup2pPairs.end()), cpup2pPairs.end());
    cpum2lPairs.erase(std::unique(cpum2lPairs.begin(), cpum2lPairs.end()), cpum2lPairs.end());

    // Build the GPU representation of the same tree.
    DeviceVector<KeyType> d_leaves = leaves;
    OctreeData<KeyType, GpuTag> gpuTree;
    gpuTree.resize(nNodes(leaves));
    buildOctreeGpu(rawPtr(d_leaves), gpuTree.data());

    // Allocate a device buffer large enough to hold all possible pairs.
    const unsigned numP2PPairs = static_cast<unsigned>(cpup2pPairs.size());
    const unsigned numM2LPairs = static_cast<unsigned>(cpum2lPairs.size());

    DeviceVector<util::array<TreeNodeIndex, 2>> d_p2pPairs(numP2PPairs);
    DeviceVector<util::array<TreeNodeIndex, 2>> d_m2lPairs(numM2LPairs);

    unsigned* d_p2pCount;
    unsigned* d_m2lCount;
    cudaMalloc(&d_p2pCount, sizeof(unsigned));
    cudaMalloc(&d_m2lCount, sizeof(unsigned));
    cudaMemset(d_p2pCount, 0, sizeof(unsigned));
    cudaMemset(d_m2lCount, 0, sizeof(unsigned));

    dim3 block(numThreadsPerBlock,1,1);
    dim3 grid(8,1,1);
    dualTraversalCount<consumerMultiple><<<grid, block>>>(
        rawPtr(gpuTree.childOffsets), 0, 0, rawPtr(d_p2pPairs), rawPtr(d_m2lPairs), d_p2pCount, d_m2lCount);

    cudaDeviceSynchronize();

    // Copy the count and pairs back to the host.
    unsigned h_p2pCount = 0;
    unsigned h_m2lCount = 0;

    cudaMemcpy(&h_p2pCount, d_p2pCount, sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_m2lCount, d_m2lCount, sizeof(unsigned), cudaMemcpyDeviceToHost);

    std::vector<util::array<TreeNodeIndex, 2>> h_p2pPairs;
    h_p2pPairs.resize(h_p2pCount);
    cudaMemcpy(h_p2pPairs.data(), rawPtr(d_p2pPairs), h_p2pCount * sizeof(util::array<TreeNodeIndex, 2>),
               cudaMemcpyDeviceToHost);
    std::sort(h_p2pPairs.begin(), h_p2pPairs.end());
    h_p2pPairs.erase(std::unique(h_p2pPairs.begin(), h_p2pPairs.end()), h_p2pPairs.end());

    std::vector<util::array<TreeNodeIndex, 2>> h_m2lPairs;
    h_m2lPairs.resize(h_m2lCount);
    cudaMemcpy(h_m2lPairs.data(), rawPtr(d_m2lPairs), h_m2lCount * sizeof(util::array<TreeNodeIndex, 2>),
               cudaMemcpyDeviceToHost);
    std::sort(h_m2lPairs.begin(), h_m2lPairs.end());
    h_m2lPairs.erase(std::unique(h_m2lPairs.begin(), h_m2lPairs.end()), h_m2lPairs.end());

    // Compare GPU results against CPU reference.
    EXPECT_EQ(h_p2pPairs.size(), cpup2pPairs.size());
    for (size_t i = 0; i < cpup2pPairs.size(); ++i)
    {
        EXPECT_EQ(h_p2pPairs[i], cpup2pPairs[i]);
    }

    EXPECT_EQ(h_m2lPairs.size(), cpum2lPairs.size());
    for (size_t i = 0; i < cpum2lPairs.size(); ++i)
    {
        EXPECT_EQ(h_m2lPairs[i], cpum2lPairs[i]);
    }


    cudaFree(d_p2pCount);
    cudaFree(d_m2lCount);
}

TEST(Traversal, dualTraversalAllPairsGpu)
{
    // Run the GPU dual traversal for both 32‑bit and 64‑bit octree keys.
    dualTraversalAllPairsGpu<unsigned>();
    dualTraversalAllPairsGpu<uint64_t>();
}

} // namespace cstone