/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Halo detection: GPU dual traversal vs. findHalosGpu benchmark and correctness test
 *
 * @author Timo Schwab <tischwab@ethz.ch>
 *
 * Compares three implementations of halo detection:
 *   1. CPU reference: findHalos (OpenMP, one singleTraversal per local leaf)
 *   2. GPU reference: findHalosGpu (one CUDA thread per local leaf)
 *   3. New GPU:       dualTraversalGPU-based kernel (cluster-based dual traversal)
 *
 * All three produce a flag array over tree nodes; we compare at leaf-node granularity.
 */

#include <iostream>
#include <vector>
#include <algorithm>

#include "gtest/gtest.h"

#include "cstone/cuda/device_vector.h"
#include "cstone/sfc/box.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/octree_gpu.h"
#include "cstone/tree/cs_util.hpp"
#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/traversal/collisions.hpp"
#include "cstone/traversal/collisions_gpu.h"
#include "cstone/traversal/traversal_gpu.cuh"

#include "coord_samples/random.hpp"
#include "../../performance/timing.cuh"

namespace cstone
{

using HaloTravConfig = TraversalConfig<1024>;

/*! @brief GPU dual-traversal kernel for halo detection */
template <int numWarps, class KeyType, class T>
__global__ void dualTraversalHalosKernel(
    const TreeNodeIndex* __restrict__ childOffsets,
    const KeyType*       __restrict__ codeStarts,
    const KeyType*       __restrict__ codeEnds,
    const Vec3<T>*       __restrict__ nodeCenters,
    const Vec3<T>*       __restrict__ nodeSizes,
    T haloFactor,
    KeyType localStart,
    KeyType localEnd,
    Box<T> box,
    TreeNodeIndex rootA,
    TreeNodeIndex rootB,
    uint8_t* __restrict__ collisionFlags,
    GlobalWorkQueue globalQueue,
    GlobalTraversalQueue globalTraversalQueue,
    unsigned* __restrict__ numActiveProducers)
{
    //! criterion: a overlaps local range, b not fully local, a's inflated box overlaps b's box
    auto criterion =
        [localStart, localEnd, codeStarts, codeEnds, nodeCenters, nodeSizes, haloFactor, box]
        __device__(TreeNodeIndex a, TreeNodeIndex b)
    {
        bool aOverlapsLocal = overlapTwoRanges(localStart, localEnd, codeStarts[a], codeEnds[a]);
        bool bNotFullyLocal = !containedIn(codeStarts[b], codeEnds[b], localStart, localEnd);
        if (!aOverlapsLocal || !bNotFullyLocal) { return false; }
        return overlap(nodeCenters[a], nodeSizes[a] * haloFactor, nodeCenters[b], nodeSizes[b], box);
    };

    auto m2l = [] __device__(TreeNodeIndex, TreeNodeIndex) {};

    //! p2p: both leaves, criterion passed — mark b as a halo
    auto p2p = [collisionFlags] __device__(TreeNodeIndex /*a*/, TreeNodeIndex b)
    {
        collisionFlags[b] = 1;
    };

    dualTraversalGPU<numWarps, HaloTravConfig>(
        childOffsets, rootA, rootB,
        globalQueue, globalTraversalQueue, numActiveProducers,
        criterion, m2l, p2p);
}

struct DualHaloConfig
{
    static constexpr unsigned numWarps = 4;

    static constexpr unsigned numThreadsPerBlock = numWarps * GpuConfig::warpSize;
    // static_assert(numThreadsPerBlock >= 64 && numThreadsPerBlock <= 512);

    //! @brief number of blocks per thread block cluster
    static constexpr unsigned kBlocksPerCluster = 8;

    //! @brief number of clusters in the grid
    static constexpr unsigned kNumClusters = 64;

    //! @brief total number of blocks launched
    static constexpr unsigned kTotalBlocks = kBlocksPerCluster * kNumClusters;
    static_assert(kBlocksPerCluster > 0 && kNumClusters > 0);

    //! @brief per-block shared memory capacity (node-pair slots)
    static constexpr unsigned stackCap = HaloTravConfig::stackCap;
};

void haloDetectionGpuTest(unsigned numParticles = 2000000, unsigned bucketSize = 16)
{
    using KeyType = uint64_t;
    using T       = double;
    Box<T> box{-1, 1};

    // ── Build leaf-level cornerstone tree ────────────────────────
    RandomGaussianCoordinates<T, MortonKey<KeyType>> randomBox(numParticles, box);
    auto particleKeys = randomBox.particleKeys();
    std::sort(particleKeys.begin(), particleKeys.end());

    std::vector<KeyType> leaves{0, nodeRange<KeyType>(0)};
    std::vector<unsigned> counts{numParticles};
    while (!updateOctree(std::span<const KeyType>(particleKeys.data(), numParticles),
                         bucketSize, leaves, counts))
        ;

    // CPU fully-linked octree
    Octree<KeyType> octree;
    octree.update(leaves.data(), nNodes(leaves));

    const TreeNodeIndex numTreeNodes = octree.numTreeNodes();
    const unsigned      numLeaves    = nNodes(leaves);

    printf("Halo test: %u particles, %u leaves, %d total nodes, bucket %u\n",
           numParticles, numLeaves, numTreeNodes, bucketSize);

    // ── Local range: first half of leaf array ────────────────────
    unsigned firstNode = 0;
    unsigned lastNode  = numLeaves / 2;
    KeyType  localStart = leaves[firstNode];
    KeyType  localEnd   = leaves[lastNode];

    printf("Local range: [%llu, %llu), leaves [%u, %u)\n",
           (unsigned long long)localStart, (unsigned long long)localEnd, firstNode, lastNode);

    // ── Per-node geometry arrays (CPU) ───────────────────────────
    std::vector<KeyType> h_codeStarts(numTreeNodes);
    std::vector<KeyType> h_codeEnds(numTreeNodes);
    std::vector<Vec3<T>> h_nodeCenters(numTreeNodes);
    std::vector<Vec3<T>> h_nodeSizes(numTreeNodes);

    for (TreeNodeIndex i = 0; i < numTreeNodes; ++i)
    {
        h_codeStarts[i]  = octree.codeStart(i);
        h_codeEnds[i]    = octree.codeEnd(i);
        IBox ibox         = sfcIBox(sfcKey(octree.codeStart(i)), octree.level(i));
        auto [c, s]       = centerAndSize<KeyType>(ibox, box);
        h_nodeCenters[i]  = c;
        h_nodeSizes[i]    = s;
    }

    constexpr T haloFactor = 2;
    std::vector<Vec3<T>> h_searchCenters(numLeaves);
    std::vector<Vec3<T>> h_searchSizes(numLeaves);
    for (unsigned leafIdx = 0; leafIdx < numLeaves; ++leafIdx)
    {
        TreeNodeIndex nodeIdx    = octree.toInternal(leafIdx);
        h_searchCenters[leafIdx] = h_nodeCenters[nodeIdx];
        h_searchSizes[leafIdx]   = h_nodeSizes[nodeIdx] * haloFactor;
    }

    // ── CPU reference: findHalos (OMP-parallel, singleTraversal per leaf) ──
    std::vector<uint8_t> h_cpuFlags(numTreeNodes, 0);
    float cpuTime = timeCpu([&]()
    {
        findHalos(octree.nodeKeys().data(),
                  octree.childOffsets().data(),
                  octree.parents().data(),
                  h_nodeCenters.data(),
                  h_nodeSizes.data(),
                  leaves.data(),
                  h_searchCenters.data(),
                  h_searchSizes.data(),
                  box,
                  firstNode, lastNode,
                  h_cpuFlags.data());
    });
    printf("CPU findHalos:         %.3f s\n", cpuTime);

    // ── Upload data to GPU ────────────────────────────────────────
    DeviceVector<KeyType>  d_leaves(leaves);
    OctreeData<KeyType, GpuTag> gpuTree;
    gpuTree.resize(numLeaves);
    buildOctreeGpu(rawPtr(d_leaves), gpuTree.data());

    DeviceVector<KeyType>       d_codeStarts(h_codeStarts);
    DeviceVector<KeyType>       d_codeEnds(h_codeEnds);
    DeviceVector<Vec3<T>>       d_nodeCenters(h_nodeCenters);
    DeviceVector<Vec3<T>>       d_nodeSizes(h_nodeSizes);
    DeviceVector<Vec3<T>>       d_searchCenters(h_searchCenters);
    DeviceVector<Vec3<T>>       d_searchSizes(h_searchSizes);

    // ── GPU reference: findHalosGpu (one thread per local leaf) ──
    uint8_t* d_gpuFlags = nullptr;
    cudaMalloc(&d_gpuFlags, numTreeNodes * sizeof(uint8_t));

    auto runFindHalosGpu = [&]()
    {
        cudaMemset(d_gpuFlags, 0, numTreeNodes * sizeof(uint8_t));
        findHalosGpu(rawPtr(gpuTree.prefixes),
                     rawPtr(gpuTree.childOffsets),
                     rawPtr(gpuTree.parents),
                     rawPtr(d_nodeCenters),
                     rawPtr(d_nodeSizes),
                     rawPtr(d_leaves),
                     rawPtr(d_searchCenters),
                     rawPtr(d_searchSizes),
                     box,
                     firstNode, lastNode,
                     d_gpuFlags);
    };

    float gpuRefTime = timeGpu(runFindHalosGpu);
    printf("GPU findHalosGpu:      %.3f ms\n", gpuRefTime);

    // ── New GPU: dualTraversalGPU-based halo detection ────────────
    uint8_t* d_dualFlags = nullptr;
    cudaMalloc(&d_dualFlags, numTreeNodes * sizeof(uint8_t));

    // ── Allocate global work buffer ──
    constexpr unsigned gChunkSize   = HaloTravConfig::chunkSize;
    constexpr unsigned gNumSegments = 1024;
    constexpr unsigned gCapacity    = gNumSegments * gChunkSize;

    TreeNodeIndex* d_gNodeA;  cudaMalloc(&d_gNodeA, gCapacity * sizeof(TreeNodeIndex));
    TreeNodeIndex* d_gNodeB;  cudaMalloc(&d_gNodeB, gCapacity * sizeof(TreeNodeIndex));
    int*           d_gIsP2P;  cudaMalloc(&d_gIsP2P, gCapacity * sizeof(int));
    unsigned*      d_writeHead;  cudaMalloc(&d_writeHead, sizeof(unsigned));
    unsigned*      d_readHead;   cudaMalloc(&d_readHead, sizeof(unsigned));
    unsigned*      d_segReady;   cudaMalloc(&d_segReady, gNumSegments * sizeof(unsigned));
    unsigned*      d_numProducers; cudaMalloc(&d_numProducers, sizeof(unsigned));

    GlobalWorkQueue gq{d_gNodeA, d_gNodeB, d_gIsP2P, d_writeHead, d_readHead, d_segReady, gNumSegments};

    // Global traversal work buffer
    constexpr unsigned tChunkSize   = HaloTravConfig::travChunkSize;
    constexpr unsigned tNumSegments = 1024;
    constexpr unsigned tCapacity    = tNumSegments * tChunkSize;

    TreeNodeIndex* d_tNodeA;  cudaMalloc(&d_tNodeA, tCapacity * sizeof(TreeNodeIndex));
    TreeNodeIndex* d_tNodeB;  cudaMalloc(&d_tNodeB, tCapacity * sizeof(TreeNodeIndex));
    unsigned*      d_tWriteHead;  cudaMalloc(&d_tWriteHead, sizeof(unsigned));
    unsigned*      d_tReadHead;   cudaMalloc(&d_tReadHead, sizeof(unsigned));
    unsigned*      d_tSegReady;   cudaMalloc(&d_tSegReady, tNumSegments * sizeof(unsigned));

    GlobalTraversalQueue tq{d_tNodeA, d_tNodeB, d_tWriteHead, d_tReadHead, d_tSegReady, tNumSegments};

    unsigned maxBlocks = maxConcurrentBlocks(
        dualTraversalHalosKernel<DualHaloConfig::numWarps, KeyType, T>,
        DualHaloConfig::numThreadsPerBlock, DualHaloConfig::kBlocksPerCluster);
    unsigned totalBlocks = std::min(DualHaloConfig::kTotalBlocks, maxBlocks);

    cudaLaunchConfig_t dualCfg{};
    dualCfg.gridDim  = {totalBlocks, 1, 1};
    dualCfg.blockDim = {DualHaloConfig::numThreadsPerBlock, 1, 1};

    cudaLaunchAttribute dualAttr{};
    dualAttr.id               = cudaLaunchAttributeClusterDimension;
    dualAttr.val.clusterDim.x = DualHaloConfig::kBlocksPerCluster;
    dualAttr.val.clusterDim.y = 1;
    dualAttr.val.clusterDim.z = 1;

    dualCfg.attrs    = &dualAttr;
    dualCfg.numAttrs = 1;

    auto runDualTraversal = [&]()
    {
        cudaMemset(d_dualFlags, 0, numTreeNodes * sizeof(uint8_t));
        cudaMemset(d_writeHead, 0, sizeof(unsigned));
        cudaMemset(d_readHead, 0, sizeof(unsigned));
        cudaMemset(d_segReady, 0, gNumSegments * sizeof(unsigned));
        cudaMemset(d_tWriteHead, 0, sizeof(unsigned));
        cudaMemset(d_tReadHead, 0, sizeof(unsigned));
        cudaMemset(d_tSegReady, 0, tNumSegments * sizeof(unsigned));
        cudaMemcpy(d_numProducers, &totalBlocks, sizeof(unsigned), cudaMemcpyHostToDevice);

        cudaLaunchKernelEx(&dualCfg,
                           dualTraversalHalosKernel<DualHaloConfig::numWarps, KeyType, T>,
                           rawPtr(gpuTree.childOffsets),
                           rawPtr(d_codeStarts),
                           rawPtr(d_codeEnds),
                           rawPtr(d_nodeCenters),
                           rawPtr(d_nodeSizes),
                           T(haloFactor),
                           localStart,
                           localEnd,
                           box,
                           TreeNodeIndex{0},
                           TreeNodeIndex{0},
                           d_dualFlags,
                           gq,
                           tq,
                           d_numProducers);
    };

    float dualTime = timeGpu(runDualTraversal);
    printf("GPU dualTraversal:     %.3f ms\n", dualTime);
    printf("Speedup (findHalosGpu/dual): %.2fx\n", gpuRefTime / dualTime);

    // ── Copy GPU flags back to host ───────────────────────────────
    std::vector<uint8_t> h_gpuFlags(numTreeNodes, 0);
    cudaMemcpy(h_gpuFlags.data(), d_gpuFlags, numTreeNodes * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    std::vector<uint8_t> h_dualFlags(numTreeNodes, 0);
    cudaMemcpy(h_dualFlags.data(), d_dualFlags, numTreeNodes * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // ── Validate CPU vs GPU reference at leaf level ───────────────
    size_t cpuGpuMismatches = 0;
    for (TreeNodeIndex i = 0; i < numTreeNodes; ++i)
    {
        if (octree.isLeaf(i) && h_cpuFlags[i] != h_gpuFlags[i])
        {
            if (cpuGpuMismatches < 5)
            {
                printf("  CPU/GPU leaf mismatch at node %d: cpu=%d gpu=%d\n",
                       i, h_cpuFlags[i], h_gpuFlags[i]);
            }
            ++cpuGpuMismatches;
        }
    }
    EXPECT_EQ(cpuGpuMismatches, 0u)
        << "CPU findHalos vs GPU findHalosGpu leaf mismatches: " << cpuGpuMismatches;

    // ── Validate dual traversal vs CPU reference at leaf level ────
    size_t dualMismatches = 0;
    for (TreeNodeIndex i = 0; i < numTreeNodes; ++i)
    {
        if (!octree.isLeaf(i)) { continue; }
        bool isLocal = containedIn(h_codeStarts[i], h_codeEnds[i], localStart, localEnd);
        if (isLocal) { continue; }

        if (h_cpuFlags[i] != h_dualFlags[i])
        {
            if (dualMismatches < 5)
            {
                printf("  CPU/dual leaf mismatch at node %d: cpu=%d dual=%d "
                       "[%llu, %llu)\n",
                       i, h_cpuFlags[i], h_dualFlags[i],
                       (unsigned long long)h_codeStarts[i],
                       (unsigned long long)h_codeEnds[i]);
            }
            ++dualMismatches;
        }
    }
    EXPECT_EQ(dualMismatches, 0u)
        << "CPU findHalos vs GPU dualTraversal leaf mismatches (external leaves): "
        << dualMismatches;

    // ── Count flagged external leaf nodes for informational output ─
    unsigned cpuFlaggedLeaves  = 0;
    unsigned gpuFlaggedLeaves  = 0;
    unsigned dualFlaggedLeaves = 0;
    for (TreeNodeIndex i = 0; i < numTreeNodes; ++i)
    {
        if (!octree.isLeaf(i)) { continue; }
        bool isLocal = containedIn(h_codeStarts[i], h_codeEnds[i], localStart, localEnd);
        if (isLocal) { continue; }
        if (h_cpuFlags[i])  { ++cpuFlaggedLeaves;  }
        if (h_gpuFlags[i])  { ++gpuFlaggedLeaves;  }
        if (h_dualFlags[i]) { ++dualFlaggedLeaves; }
    }
    printf("Flagged external leaf nodes — CPU: %u, findHalosGpu: %u, dual: %u\n",
           cpuFlaggedLeaves, gpuFlaggedLeaves, dualFlaggedLeaves);

    EXPECT_GT(cpuFlaggedLeaves, 0u) << "No external leaves flagged — test is vacuous";

    // ── Sanity check: global buffer was used ──
    unsigned h_wHead = 0, h_rHead = 0;
    cudaMemcpy(&h_wHead, d_writeHead, sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_rHead, d_readHead, sizeof(unsigned), cudaMemcpyDeviceToHost);
    printf("Global buffer: writeHead=%u, readHead=%u\n", h_wHead, h_rHead);

    cudaFree(d_gpuFlags);
    cudaFree(d_dualFlags);
    cudaFree(d_gNodeA);
    cudaFree(d_gNodeB);
    cudaFree(d_gIsP2P);
    cudaFree(d_writeHead);
    cudaFree(d_readHead);
    cudaFree(d_segReady);
    cudaFree(d_numProducers);
    cudaFree(d_tNodeA);
    cudaFree(d_tNodeB);
    cudaFree(d_tWriteHead);
    cudaFree(d_tReadHead);
    cudaFree(d_tSegReady);
}

TEST(Traversal, haloDetectionGpu)
{
    haloDetectionGpuTest(2000000, 16);
}

} // namespace cstone
