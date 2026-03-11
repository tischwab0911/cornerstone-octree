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

using DefaultTravConfig = TraversalConfig<1024, 256, 512, 384, 128, 768, 128, 512, 384, 16>;

template<int numWarps>
__global__ void dualTraversalCount(const TreeNodeIndex* __restrict__ childOffsets,
                                        TreeNodeIndex rootA,
                                        TreeNodeIndex rootB,
                                        util::array<TreeNodeIndex, 2>* p2pPairs,
                                        util::array<TreeNodeIndex, 2>* m2lPairs,
                                        unsigned* p2pPairCount,
                                        unsigned* m2lPairCount,
                                        GlobalWorkQueue globalQueue,
                                        GlobalTraversalQueue globalTraversalQueue,
                                        unsigned* numActiveProducers)
{
    // admissibility criterion: accept all internal node pairs
    auto allPairs = [] __device__(TreeNodeIndex, TreeNodeIndex) { return true; };

    // multipole-to-local interaction
    auto m2l = [m2lPairs, m2lPairCount] __device__(TreeNodeIndex a, TreeNodeIndex b) {
        unsigned idx = atomicAdd(m2lPairCount, 1u);
        m2lPairs[idx][0] = a;
        m2lPairs[idx][1] = b;
    };

    // particle-to-particle interaction: record each leaf pair atomically
    auto p2p = [p2pPairs, p2pPairCount] __device__(TreeNodeIndex a, TreeNodeIndex b) {
        unsigned idx = atomicAdd(p2pPairCount, 1u);
        p2pPairs[idx][0] = a;
        p2pPairs[idx][1] = b;
    };

    dualTraversalGPU<numWarps, DefaultTravConfig>(
        childOffsets, rootA, rootB,
        globalQueue, globalTraversalQueue, numActiveProducers,
        allPairs, m2l, p2p);
}

struct TravConfig {

    static constexpr unsigned numWarps = 4;

    static constexpr unsigned numThreadsPerBlock = numWarps * GpuConfig::warpSize;
    static_assert(numThreadsPerBlock >= 64 && numThreadsPerBlock <= 512);

    //! @brief number of blocks per thread block cluster
    static constexpr unsigned kBlocksPerCluster = 8;

    //! @brief number of clusters in the grid
    static constexpr unsigned kNumClusters      = 8;

    //! @brief total number of blocks launched in the grid
    static constexpr unsigned kTotalBlocks      = kBlocksPerCluster * kNumClusters;
    static_assert(kBlocksPerCluster > 0 && kNumClusters > 0);

    //! @brief per-block shared memory capacity (node-pair slots)
    static constexpr unsigned stackCap = DefaultTravConfig::stackCap;
};

template <class KeyType>
void dualTraversalAllPairsGpu()
{
    Octree<KeyType> cpuTree;
    auto leaves = OctreeMaker<KeyType>{}.divide().divide(0).divide(0, 7).divide(7).makeTree();
    cpuTree.update(leaves.data(), nNodes(leaves));

    // Compute the reference set of leaf-pairs on the CPU
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

    // Allocate device buffers
    const unsigned numP2PPairs = static_cast<unsigned>(cpup2pPairs.size());
    const unsigned numM2LPairs = static_cast<unsigned>(cpum2lPairs.size());

    DeviceVector<util::array<TreeNodeIndex, 2>> d_p2pPairs(numP2PPairs * 2);
    DeviceVector<util::array<TreeNodeIndex, 2>> d_m2lPairs(numM2LPairs * 2);

    unsigned* d_p2pCount;
    unsigned* d_m2lCount;
    cudaMalloc(&d_p2pCount, sizeof(unsigned));
    cudaMalloc(&d_m2lCount, sizeof(unsigned));
    cudaMemset(d_p2pCount, 0, sizeof(unsigned));
    cudaMemset(d_m2lCount, 0, sizeof(unsigned));

    // Global work buffer
    constexpr unsigned gChunk = DefaultTravConfig::chunkSize;
    constexpr unsigned gSegs  = 256;
    constexpr unsigned gCap   = gSegs * gChunk;
    TreeNodeIndex* d_gA; cudaMalloc(&d_gA, gCap * sizeof(TreeNodeIndex));
    TreeNodeIndex* d_gB; cudaMalloc(&d_gB, gCap * sizeof(TreeNodeIndex));
    int* d_gP2P;         cudaMalloc(&d_gP2P, gCap * sizeof(int));
    unsigned* d_wHead;   cudaMalloc(&d_wHead, sizeof(unsigned));
    unsigned* d_rHead;   cudaMalloc(&d_rHead, sizeof(unsigned));
    unsigned* d_segR;    cudaMalloc(&d_segR, gSegs * sizeof(unsigned));
    unsigned* d_nProd;   cudaMalloc(&d_nProd, sizeof(unsigned));
    cudaMemset(d_wHead, 0, sizeof(unsigned));
    cudaMemset(d_rHead, 0, sizeof(unsigned));
    cudaMemset(d_segR, 0, gSegs * sizeof(unsigned));
    unsigned maxBlocks = maxConcurrentBlocks(
        dualTraversalCount<TravConfig::numWarps>,
        TravConfig::numThreadsPerBlock, TravConfig::kBlocksPerCluster);
    unsigned totalBlocks = std::min(TravConfig::kTotalBlocks, maxBlocks);
    cudaMemcpy(d_nProd, &totalBlocks, sizeof(unsigned), cudaMemcpyHostToDevice);
    GlobalWorkQueue gq{d_gA, d_gB, d_gP2P, d_wHead, d_rHead, d_segR, gSegs};

    // Global traversal work buffer
    constexpr unsigned tChunk = DefaultTravConfig::travChunkSize;
    constexpr unsigned tSegs  = 256;
    constexpr unsigned tCap   = tSegs * tChunk;
    TreeNodeIndex* d_tA; cudaMalloc(&d_tA, tCap * sizeof(TreeNodeIndex));
    TreeNodeIndex* d_tB; cudaMalloc(&d_tB, tCap * sizeof(TreeNodeIndex));
    unsigned* d_twHead;  cudaMalloc(&d_twHead, sizeof(unsigned));
    unsigned* d_trHead;  cudaMalloc(&d_trHead, sizeof(unsigned));
    unsigned* d_tsegR;   cudaMalloc(&d_tsegR, tSegs * sizeof(unsigned));
    cudaMemset(d_twHead, 0, sizeof(unsigned));
    cudaMemset(d_trHead, 0, sizeof(unsigned));
    cudaMemset(d_tsegR, 0, tSegs * sizeof(unsigned));
    GlobalTraversalQueue tq{d_tA, d_tB, d_twHead, d_trHead, d_tsegR, tSegs};

    // Cluster launch configuration
    cudaLaunchConfig_t cfg{};
    cfg.gridDim   = {totalBlocks, 1, 1};
    cfg.blockDim  = {TravConfig::numThreadsPerBlock, 1, 1};

    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = TravConfig::kBlocksPerCluster;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;

    cfg.attrs    = &attr;
    cfg.numAttrs = 1;

    cudaLaunchKernelEx(&cfg, dualTraversalCount<TravConfig::numWarps>,
                       rawPtr(gpuTree.childOffsets), 0, 0, rawPtr(d_p2pPairs), rawPtr(d_m2lPairs), d_p2pCount, d_m2lCount, gq, tq, d_nProd);

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

    printf("Expected P2P Pairs: %zu, GPU Computed P2P Pairs: %zu \n", cpup2pPairs.size(), h_p2pPairs.size());
    printf("Expected M2L Pairs: %zu, GPU Computed M2L Pairs: %zu \n", cpum2lPairs.size(), h_m2lPairs.size());

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
    cudaFree(d_gA);
    cudaFree(d_gB);
    cudaFree(d_gP2P);
    cudaFree(d_wHead);
    cudaFree(d_rHead);
    cudaFree(d_segR);
    cudaFree(d_nProd);
    cudaFree(d_tA);
    cudaFree(d_tB);
    cudaFree(d_twHead);
    cudaFree(d_trHead);
    cudaFree(d_tsegR);
}

TEST(Traversal, dualTraversalAllPairsGpu)
{
    // Run the GPU dual traversal for both 32-bit and 64-bit octree keys.
    dualTraversalAllPairsGpu<unsigned>();
    dualTraversalAllPairsGpu<uint64_t>();
}

} // namespace cstone
