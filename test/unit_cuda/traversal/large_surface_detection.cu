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

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/focus/source_center_gpu.h"
#include "cstone/tree/octree_gpu.h"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/update_gpu.cuh"
#include "cstone/tree/cs_util.hpp"
#include "cstone/traversal/macs.hpp"
#include "cstone/traversal/traversal.hpp"
#include "cstone/traversal/traversal_gpu.cuh"
#include "cstone/traversal/collisions_gpu.h"

#include "coord_samples/random.hpp"
#include "../../performance/timing.cuh"

namespace cstone {

template <class KeyType>
__global__ void singleTraversalSurfaceCount(
    const TreeNodeIndex* __restrict__ childOffsets,
    const TreeNodeIndex* __restrict__ parents,
    const KeyType* __restrict__ codeStarts,
    const KeyType* __restrict__ codeEnds,
    const unsigned* __restrict__ nodeLevels,
    KeyType focusStart,
    KeyType focusEnd,
    Box<double> box,
    util::array<TreeNodeIndex, 2>* p2pPairs,
    unsigned* p2pPairCount)
{
    // singleTraversal is sequential — only one thread should run it
    if (blockIdx.x != 0 || threadIdx.x != 0) { return; }

    auto isSurface = [focusStart, focusEnd, codeStarts, codeEnds]
                     __device__(TreeNodeIndex i)
    {
        return overlapTwoRanges(focusStart, focusEnd, codeStarts[i], codeEnds[i]);
    };

    auto endpointAction = [p2pPairs, p2pPairCount, focusStart, focusEnd, codeStarts, codeEnds]
                          __device__(TreeNodeIndex i)
    {
        if (!containedIn(codeStarts[i], codeEnds[i], focusStart, focusEnd)) { return; }
        unsigned idx = atomicAdd(p2pPairCount, 1u);
        p2pPairs[idx][0] = i;
        p2pPairs[idx][1] = i;
    };

    singleTraversal(childOffsets, parents, isSurface, endpointAction);
}

template <int numWarps, unsigned queueCap, class KeyType>
__global__ void dualTraversalSurfaceCount(
    const TreeNodeIndex* __restrict__ childOffsets,
    const KeyType* __restrict__ codeStarts,
    const KeyType* __restrict__ codeEnds,
    const unsigned* __restrict__ nodeLevels,
    KeyType focusStart,
    KeyType focusEnd,
    Box<double> box,
    TreeNodeIndex rootA,
    TreeNodeIndex rootB,
    util::array<TreeNodeIndex, 2>* p2pPairs,
    util::array<TreeNodeIndex, 2>* m2lPairs,
    unsigned* p2pPairCount,
    unsigned* m2lPairCount)
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

    auto m2l = [m2lPairs, m2lPairCount] __device__(TreeNodeIndex a, TreeNodeIndex b) {
        unsigned idx = atomicAdd(m2lPairCount, 1u);
        m2lPairs[idx][0] = a;
        m2lPairs[idx][1] = b;
    };

    auto p2p = [p2pPairs, p2pPairCount] __device__(TreeNodeIndex a, TreeNodeIndex b) {
        unsigned idx = atomicAdd(p2pPairCount, 1u);
        p2pPairs[idx][0] = a;
        p2pPairs[idx][1] = b;
    };

    dualTraversalGPU<numWarps, queueCap>(childOffsets, rootA, rootB,
                                        crossFocusSurfacePairs, m2l, p2p);
}

struct SingleTravConfig {
    static constexpr unsigned numThreadsPerBlock   = 1;
    static constexpr unsigned kTotalBlocks         = 1;
};

struct DualTravConfig {

    static constexpr unsigned numWarps = 8;

    static constexpr unsigned numThreadsPerBlock = numWarps * GpuConfig::warpSize;
    static_assert(numThreadsPerBlock >= 64 && numThreadsPerBlock <= 512);

    //! @brief number of blocks per thread block cluster
    static constexpr unsigned kBlocksPerCluster = 8;

    //! @brief number of clusters in the grid
    static constexpr unsigned kNumClusters      = 64;

    //! @brief total number of blocks launched in the grid
    static constexpr unsigned kTotalBlocks      = kBlocksPerCluster * kNumClusters;
    static_assert(kBlocksPerCluster > 0 && kNumClusters > 0);

    //! @brief per-queue capacity
    static constexpr unsigned queueCap = numThreadsPerBlock * 4;
};

void dualVsSingleTraversalSurfaceGpu(unsigned numParticles = 2000000,
                                      unsigned bucketSize   = 16)
{
    using KeyType = uint64_t;
    using T       = double;
    Box<T> box{-1, 1};

    // ── Build tree on CPU from Gaussian particles ────────────────
    RandomGaussianCoordinates<T, MortonKey<KeyType>> randomBox(numParticles, box);

    auto particleKeys = randomBox.particleKeys();
    std::sort(particleKeys.begin(), particleKeys.end());

    std::vector<KeyType> leaves{0, nodeRange<KeyType>(0)};
    std::vector<unsigned> counts{numParticles};
    while (!updateOctree(std::span<const KeyType>(particleKeys.data(), numParticles),
                         bucketSize, leaves, counts))
        ;

    Octree<KeyType> octree;
    octree.update(leaves.data(), nNodes(leaves));

    const TreeNodeIndex numTreeNodes = octree.numTreeNodes();
    const unsigned numLeaves = nNodes(leaves);

    printf("Tree: %u particles, %u leaves, %d total nodes, bucket %u\n",
           numParticles, numLeaves, numTreeNodes, bucketSize);

    // ── Extract per-node arrays from CPU tree ────────────────────
    std::vector<KeyType>       h_codeStarts(numTreeNodes);
    std::vector<KeyType>       h_codeEnds(numTreeNodes);
    std::vector<unsigned>      h_levels(numTreeNodes);

    for (TreeNodeIndex i = 0; i < numTreeNodes; ++i)
    {
        h_codeStarts[i] = octree.codeStart(i);
        h_codeEnds[i]   = octree.codeEnd(i);
        h_levels[i]     = octree.level(i);
    }

    // ── Upload to GPU ────────────────────────────────────────────
    auto co = octree.childOffsets();
    auto pa = octree.parents();
    DeviceVector<TreeNodeIndex> d_childOffsets(co.data(), co.data() + co.size());
    DeviceVector<TreeNodeIndex> d_parents(pa.data(), pa.data() + pa.size());
    DeviceVector<KeyType>       d_codeStarts(h_codeStarts);
    DeviceVector<KeyType>       d_codeEnds(h_codeEnds);
    DeviceVector<unsigned>      d_levels(h_levels);

    KeyType focusStart = 0;
    KeyType focusEnd   = nodeRange<KeyType>(0) / 8;

    printf("Focus range: [%llu, %llu) — first octant\n",
           (unsigned long long)focusStart, (unsigned long long)focusEnd);

    // ── CPU reference: dual traversal ────────────────────────────
    std::vector<util::array<TreeNodeIndex, 2>> cpuPairs;
    {
        auto crossFocusCPU = [focusStart, focusEnd, &octree, &box](TreeNodeIndex a, TreeNodeIndex b)
        {
            bool aFocusOverlap = overlapTwoRanges(focusStart, focusEnd, octree.codeStart(a), octree.codeEnd(a));
            bool bInFocus      = containedIn(octree.codeStart(b), octree.codeEnd(b), focusStart, focusEnd);
            if (!aFocusOverlap || bInFocus) { return false; }

            IBox aBox = sfcIBox(sfcKey(octree.codeStart(a)), octree.level(a));
            IBox bBox = sfcIBox(sfcKey(octree.codeStart(b)), octree.level(b));
            return minDistanceSq<KeyType>(aBox, bBox, box) == 0.0;
        };
        auto m2lNoop = [](TreeNodeIndex, TreeNodeIndex) {};
        auto p2pCpu  = [&cpuPairs](TreeNodeIndex a, TreeNodeIndex b) { cpuPairs.push_back({a, b}); };

        dualTraversal(octree.childOffsets().data(), 0, 0, crossFocusCPU, m2lNoop, p2pCpu);
        std::sort(cpuPairs.begin(), cpuPairs.end());
    }
    printf("CPU reference pairs: %zu\n", cpuPairs.size());

    // ── Allocate output buffers ──────────────────────────────────
    const unsigned pairBufSize = 1u << 24; // 16M pairs

    DeviceVector<util::array<TreeNodeIndex, 2>> d_singleP2P(pairBufSize);
    DeviceVector<util::array<TreeNodeIndex, 2>> d_dualP2P(pairBufSize);
    DeviceVector<util::array<TreeNodeIndex, 2>> d_dualM2L(pairBufSize);

    unsigned* d_singleP2PCount;
    unsigned* d_dualP2PCount;
    unsigned* d_dualM2LCount;
    cudaMalloc(&d_singleP2PCount, sizeof(unsigned));
    cudaMalloc(&d_dualP2PCount,   sizeof(unsigned));
    cudaMalloc(&d_dualM2LCount,   sizeof(unsigned));

    // ── Dual traversal launch configuration (cluster-based) ──────
    // unsigned smemBytes = dualTraversalSmemBytes(DualTravConfig::queueCap, DualTravConfig::numWarps);

    cudaLaunchConfig_t dualCfg{};
    dualCfg.gridDim  = {DualTravConfig::kTotalBlocks, 1, 1};
    dualCfg.blockDim = {DualTravConfig::numThreadsPerBlock, 1, 1};
    // dualCfg.dynamicSmemBytes = smemBytes;

    cudaLaunchAttribute dualAttr{};
    dualAttr.id               = cudaLaunchAttributeClusterDimension;
    dualAttr.val.clusterDim.x = DualTravConfig::kBlocksPerCluster;
    dualAttr.val.clusterDim.y = 1;
    dualAttr.val.clusterDim.z = 1;

    dualCfg.attrs    = &dualAttr;
    dualCfg.numAttrs = 1;

    // ══════════════════════════════════════════════════════════════
    //  SINGLE TRAVERSAL (GPU)
    // ══════════════════════════════════════════════════════════════
    auto runSingle = [&]()
    {
        cudaMemset(d_singleP2PCount, 0, sizeof(unsigned));
        singleTraversalSurfaceCount<KeyType>
            <<<SingleTravConfig::kTotalBlocks, SingleTravConfig::numThreadsPerBlock>>>(
                rawPtr(d_childOffsets),
                rawPtr(d_parents),
                rawPtr(d_codeStarts),
                rawPtr(d_codeEnds),
                rawPtr(d_levels),
                focusStart,
                focusEnd,
                box,
                rawPtr(d_singleP2P),
                d_singleP2PCount);
    };

    float singleTime = timeGpu(runSingle);
    printf("Single traversal: %.3f ms\n", singleTime);

    // ══════════════════════════════════════════════════════════════
    //  DUAL TRAVERSAL (GPU)
    // ══════════════════════════════════════════════════════════════
    auto runDual = [&]()
    {
        cudaMemset(d_dualP2PCount, 0, sizeof(unsigned));
        cudaMemset(d_dualM2LCount, 0, sizeof(unsigned));
        cudaLaunchKernelEx(&dualCfg,
                           dualTraversalSurfaceCount<DualTravConfig::numWarps, DualTravConfig::queueCap, KeyType>,
                           rawPtr(d_childOffsets),
                           rawPtr(d_codeStarts),
                           rawPtr(d_codeEnds),
                           rawPtr(d_levels),
                           focusStart,
                           focusEnd,
                           box,
                           0, 0,
                           rawPtr(d_dualP2P),
                           rawPtr(d_dualM2L),
                           d_dualP2PCount,
                           d_dualM2LCount);
    };

    float dualTime = timeGpu(runDual);
    printf("Dual traversal:   %.3f ms\n", dualTime);
    printf("Speedup (single/dual): %.2fx\n", singleTime / dualTime);

    // ── Copy counts ──────────────────────────────────────────────
    unsigned h_singleCount = 0, h_dualP2PCount = 0, h_dualM2LCount = 0;

    cudaMemcpy(&h_singleCount,  d_singleP2PCount, sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_dualP2PCount, d_dualP2PCount,   sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_dualM2LCount, d_dualM2LCount,   sizeof(unsigned), cudaMemcpyDeviceToHost);

    printf("Single P2P pairs: %u\n", h_singleCount);
    printf("Dual   P2P pairs: %u, M2L pairs: %u\n", h_dualP2PCount, h_dualM2LCount);

    ASSERT_LE(h_singleCount,  pairBufSize) << "Single traversal overflowed";
    ASSERT_LE(h_dualP2PCount, pairBufSize) << "Dual P2P overflowed";
    ASSERT_LE(h_dualM2LCount, pairBufSize) << "Dual M2L overflowed";

    // ── Compare GPU dual pairs against CPU reference ─────────────
    std::vector<util::array<TreeNodeIndex, 2>> h_dualPairs(h_dualP2PCount);

    cudaMemcpy(h_dualPairs.data(), rawPtr(d_dualP2P),
               h_dualP2PCount * sizeof(util::array<TreeNodeIndex, 2>),
               cudaMemcpyDeviceToHost);

    std::sort(h_dualPairs.begin(), h_dualPairs.end());
    h_dualPairs.erase(std::unique(h_dualPairs.begin(), h_dualPairs.end()), h_dualPairs.end());

    printf("GPU dual pairs (unique): %zu, CPU reference: %zu\n",
           h_dualPairs.size(), cpuPairs.size());

    EXPECT_EQ(h_dualPairs.size(), cpuPairs.size());

    size_t mismatches = 0;
    for (size_t i = 0; i < std::min(h_dualPairs.size(), cpuPairs.size()); ++i)
    {
        if (h_dualPairs[i] != cpuPairs[i])
        {
            if (mismatches < 10)
                printf("  Mismatch at %zu: gpu=(%d,%d) vs cpu=(%d,%d)\n", i,
                       h_dualPairs[i][0], h_dualPairs[i][1],
                       cpuPairs[i][0], cpuPairs[i][1]);
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u) << "Total mismatches: " << mismatches;

    // ── Validate pair properties ─────────────────────────────────
    size_t propertyErrors = 0;
    for (auto& p : h_dualPairs)
    {
        TreeNodeIndex a = p[0], b = p[1];
        bool aInFocus = h_codeStarts[a] >= focusStart && h_codeEnds[a] <= focusEnd;
        bool bOutside = h_codeStarts[b] >= focusEnd || h_codeEnds[b] <= focusStart;

        IBox aBox = sfcIBox(sfcKey(h_codeStarts[a]), h_levels[a]);
        IBox bBox = sfcIBox(sfcKey(h_codeStarts[b]), h_levels[b]);
        bool touching = minDistanceSq<KeyType>(aBox, bBox, box) == 0.0;

        if (!aInFocus || !bOutside || !touching)
        {
            if (propertyErrors < 10)
            {
                printf("  Bad pair (%d,%d): aInFocus=%d bOutside=%d touching=%d\n",
                       a, b, aInFocus, bOutside, touching);
            }
            ++propertyErrors;
        }
    }
    EXPECT_EQ(propertyErrors, 0u) << "Pairs violating surface properties: " << propertyErrors;

    // ── Cleanup ──────────────────────────────────────────────────
    cudaFree(d_singleP2PCount);
    cudaFree(d_dualP2PCount);
    cudaFree(d_dualM2LCount);
}

TEST(Traversal, dualVsSingleSurfaceGpu)
{
    dualVsSingleTraversalSurfaceGpu(2000000, 16);
}

}
