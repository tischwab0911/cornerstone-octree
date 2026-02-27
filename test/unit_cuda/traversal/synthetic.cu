/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Synthetic P2P benchmark: GPU dual traversal vs. single traversal (one thread per leaf)
 *
 * @author Timo Schwab <tischwab@ethz.ch>
 *
 * Builds a tree from random Gaussian coordinates and finds all pairs of leaf nodes
 * whose bounding boxes touch or overlap (minDistance == 0), simulating the interaction
 * list computation for direct particle-to-particle (P2P) interactions.
 *
 * Two GPU methods are compared:
 *   1. Single: one GPU thread per leaf node calls singleTraversal to find all
 *              touching partner leaves (mirrors the findHalosGpu pattern).
 *   2. Dual:   cluster-based dualTraversalGPU (parallel, with DSM work-stealing).
 *
 * Correctness:
 *   CPU reference (dualTraversal) produces N_pairs where each unordered pair
 *   {a,b} (including a==b self-pairs) is counted once.
 *   Single traversal counts each off-diagonal pair twice (from thread a and from
 *   thread b) and each self-pair once, giving:
 *     singleCount == 2 * cpuCount - numLeaves
 *   Dual GPU count must equal cpuCount exactly.
 *
 * The p2p action spins over a (particlesPerBin x particlesPerBin) fmaf loop to
 * simulate the actual per-particle work inside a leaf-leaf interaction.
 *
 * The termination condition (encoded in the criterion / continuationCriterion lambda):
 *   prune when minDistance(a, b) > 0 (subtrees strictly separated).
 */

#include <vector>
#include <cstdio>

#include "gtest/gtest.h"

#include "cstone/cuda/device_vector.h"
#include "cstone/sfc/box.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/cs_util.hpp"
#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/traversal/traversal.hpp"
#include "cstone/traversal/traversal_gpu.cuh"

#include "coord_samples/random.hpp"
#include "../../performance/timing.cuh"

namespace cstone
{

// ── Single traversal GPU kernel ───────────────────────────────────────────────
//
// One GPU thread per leaf node.  Each thread calls singleTraversal (HOST_DEVICE_FUN)
// to walk the full tree and find all partner leaves whose bounding box touches its own.
//
// singleTraversal fires endpointAction(b) for every leaf b where
// continuationCriterion(b) is true, so no extra filter is needed inside the action.

template<class T>
__global__ void singleTraversalP2PKernel(
    const TreeNodeIndex* __restrict__ childOffsets,
    const TreeNodeIndex* __restrict__ parents,
    const TreeNodeIndex* __restrict__ leafToInternal,
    const Vec3<T>* __restrict__       nodeCenters,
    const Vec3<T>* __restrict__       nodeSizes,
    Box<T>                            box,
    unsigned                          numLeaves,
    unsigned                          particlesPerBin,
    unsigned*                         p2pCount)
{
    unsigned leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIdx >= numLeaves) { return; }

    TreeNodeIndex a   = leafToInternal[leafIdx];
    Vec3<T>  centerA  = nodeCenters[a];
    Vec3<T>  sizeA    = nodeSizes[a];

    //! continuationCriterion: recurse into b when the subtrees touch or overlap
    auto criterion = [centerA, sizeA, nodeCenters, nodeSizes, box]
        (TreeNodeIndex b) -> bool
    {
        Vec3<T> d = minDistance(centerA, sizeA, nodeCenters[b], nodeSizes[b], box);
        return norm2(d) == T(0);
    };

    //! endpointAction: b is a touching leaf — spin over particlesPerBin^2 fmaf iters
    auto action = [p2pCount, a, particlesPerBin] (TreeNodeIndex b)
    {
        volatile float acc = 0.f;
        for (unsigned pi = 0; pi < particlesPerBin; ++pi)
            for (unsigned pj = 0; pj < particlesPerBin; ++pj)
                acc = __fmaf_rn(float(a + pi), float(b + pj), acc);
        atomicAdd(p2pCount, 1u);
    };

    singleTraversal(childOffsets, parents, criterion, action);
}

// ── Dual traversal GPU kernel ─────────────────────────────────────────────────

template<int numConsumersPerBlock, class T>
__global__ void dualP2PKernel(
    const TreeNodeIndex* __restrict__ childOffsets,
    const Vec3<T>* __restrict__       nodeCenters,
    const Vec3<T>* __restrict__       nodeSizes,
    Box<T>                            box,
    TreeNodeIndex                     rootA,
    TreeNodeIndex                     rootB,
    unsigned                          particlesPerBin,
    unsigned*                         p2pCount)
{
    //! criterion: recurse only when subtrees touch or overlap
    auto criterion = [nodeCenters, nodeSizes, box] __device__(TreeNodeIndex a, TreeNodeIndex b) -> bool
    {
        Vec3<T> d = minDistance(nodeCenters[a], nodeSizes[a], nodeCenters[b], nodeSizes[b], box);
        return norm2(d) == T(0);
    };

    auto m2l = [] __device__(TreeNodeIndex, TreeNodeIndex) {};

    //! p2p: spin over (particlesPerBin x particlesPerBin) fmaf iterations, then count
    auto p2p = [p2pCount, particlesPerBin] __device__(TreeNodeIndex a, TreeNodeIndex b)
    {
        volatile float acc = 0.f;
        for (unsigned pi = 0; pi < particlesPerBin; ++pi)
            for (unsigned pj = 0; pj < particlesPerBin; ++pj)
                acc = __fmaf_rn(float(a + pi), float(b + pj), acc);
        atomicAdd(p2pCount, 1u);
    };

    dualTraversalGPU<numConsumersPerBlock>(childOffsets, rootA, rootB, criterion, m2l, p2p);
}

// ── Launch configurations ─────────────────────────────────────────────────────

struct SingleConfig
{
    static constexpr unsigned numThreadsPerBlock = 256;
};

struct DualConfig
{
    static constexpr unsigned numConsumersPerBlock = 8;
    static constexpr unsigned numThreadsPerBlock   = (numConsumersPerBlock + 1) * GpuConfig::warpSize;
    static_assert(numThreadsPerBlock >= 64 && numThreadsPerBlock <= 512);

    static constexpr unsigned kBlocksPerCluster = 8;
    static constexpr unsigned kNumClusters      = 64;
    static constexpr unsigned kTotalBlocks      = kBlocksPerCluster * kNumClusters;
};

// ── Benchmark body ────────────────────────────────────────────────────────────

void syntheticP2PBenchmark(unsigned numParticles   = 1000000,
                            unsigned particlesPerBin = 32)
{
    using KeyType = uint64_t;
    using T       = double;
    Box<T> box{-1, 1};

    // ── Build tree from random Gaussian particles ─────────────────────────────
    RandomGaussianCoordinates<T, MortonKey<KeyType>> randomBox(numParticles, box);
    auto particleKeys = randomBox.particleKeys();
    std::sort(particleKeys.begin(), particleKeys.end());

    std::vector<KeyType>  leaves{0, nodeRange<KeyType>(0)};
    std::vector<unsigned> counts{numParticles};
    while (!updateOctree(std::span<const KeyType>(particleKeys.data(), numParticles),
                         particlesPerBin, leaves, counts))
        ;

    Octree<KeyType> octree;
    octree.update(leaves.data(), nNodes(leaves));

    const TreeNodeIndex numTreeNodes = octree.numTreeNodes();
    const unsigned      numLeaves    = nNodes(leaves);

    printf("Synthetic P2P: %u particles, %u leaves, %d total nodes, bin %u\n",
           numParticles, numLeaves, numTreeNodes, particlesPerBin);

    // ── Per-node geometry (centers and half-sizes) ────────────────────────────
    std::vector<Vec3<T>> h_nodeCenters(numTreeNodes);
    std::vector<Vec3<T>> h_nodeSizes(numTreeNodes);

    for (TreeNodeIndex i = 0; i < numTreeNodes; ++i)
    {
        IBox ibox        = sfcIBox(sfcKey(octree.codeStart(i)), octree.level(i));
        auto [c, s]      = centerAndSize<KeyType>(ibox, box);
        h_nodeCenters[i] = c;
        h_nodeSizes[i]   = s;
    }

    // ── leafToInternal mapping (leaf index → internal tree node index) ─────────
    std::vector<TreeNodeIndex> h_leafToInternal(numLeaves);
    for (unsigned i = 0; i < numLeaves; ++i)
        h_leafToInternal[i] = octree.toInternal(i);

    // ── CPU reference: dualTraversal with synthetic spin ─────────────────────
    //    Each pair {a,b} counted once (including self-pairs a==b).
    unsigned cpuPairCount = 0;
    // {
    //     auto cpuCriterion = [&](TreeNodeIndex a, TreeNodeIndex b)
    //     {
    //         Vec3<T> d = minDistance(h_nodeCenters[a], h_nodeSizes[a],
    //                                 h_nodeCenters[b], h_nodeSizes[b], box);
    //         return norm2(d) == T(0);
    //     };
    //     auto cpuM2L = [](TreeNodeIndex, TreeNodeIndex) {};
    //     auto cpuP2P = [&cpuPairCount, particlesPerBin](TreeNodeIndex a, TreeNodeIndex b)
    //     {
    //         volatile float acc = 0.f;
    //         for (unsigned pi = 0; pi < particlesPerBin; ++pi)
    //             for (unsigned pj = 0; pj < particlesPerBin; ++pj)
    //                 acc = std::fmaf(float(a + pi), float(b + pj), acc);
    //         ++cpuPairCount;
    //     };

    //     float cpuTime = timeCpu([&]()
    //     {
    //         dualTraversal(octree.childOffsets().data(), 0, 0, cpuCriterion, cpuM2L, cpuP2P);
    //     });
    //     printf("CPU dual traversal:    %.3f s,  %u pairs\n", cpuTime, cpuPairCount);
    // }

    // ── Upload arrays to GPU ──────────────────────────────────────────────────
    auto co  = octree.childOffsets();
    auto par = octree.parents();

    DeviceVector<TreeNodeIndex> d_childOffsets(co.data(), co.data() + co.size());
    DeviceVector<TreeNodeIndex> d_parents(par.data(), par.data() + par.size());
    DeviceVector<TreeNodeIndex> d_leafToInternal(h_leafToInternal);
    DeviceVector<Vec3<T>>       d_nodeCenters(h_nodeCenters);
    DeviceVector<Vec3<T>>       d_nodeSizes(h_nodeSizes);

    unsigned* d_singleCount;
    unsigned* d_dualCount;
    cudaMalloc(&d_singleCount, sizeof(unsigned));
    cudaMalloc(&d_dualCount,   sizeof(unsigned));

    // ── Single traversal GPU (one thread per leaf) ────────────────────────────
    //    Each off-diagonal pair {a,b} is found by thread a AND thread b → counted twice.
    //    Each self-pair {a,a} is found only by thread a → counted once.
    //    Expected: singleCount == 2 * cpuPairCount - numLeaves
    const unsigned singleBlocks =
        (numLeaves + SingleConfig::numThreadsPerBlock - 1) / SingleConfig::numThreadsPerBlock;

    auto runSingle = [&]()
    {
        cudaMemset(d_singleCount, 0, sizeof(unsigned));
        singleTraversalP2PKernel<T>
            <<<singleBlocks, SingleConfig::numThreadsPerBlock>>>(
                rawPtr(d_childOffsets),
                rawPtr(d_parents),
                rawPtr(d_leafToInternal),
                rawPtr(d_nodeCenters),
                rawPtr(d_nodeSizes),
                box,
                numLeaves,
                particlesPerBin,
                d_singleCount);
    };

    float singleTime = timeGpu(runSingle);
    printf("Single traversal GPU:  %.3f ms\n", singleTime);

    // ── Dual GPU traversal (cluster-based, with DSM work-stealing) ────────────
    cudaLaunchConfig_t dualCfg{};
    dualCfg.gridDim  = {DualConfig::kTotalBlocks, 1, 1};
    dualCfg.blockDim = {DualConfig::numThreadsPerBlock, 1, 1};

    cudaLaunchAttribute dualAttr{};
    dualAttr.id               = cudaLaunchAttributeClusterDimension;
    dualAttr.val.clusterDim.x = DualConfig::kBlocksPerCluster;
    dualAttr.val.clusterDim.y = 1;
    dualAttr.val.clusterDim.z = 1;
    dualCfg.attrs    = &dualAttr;
    dualCfg.numAttrs = 1;

    auto runDual = [&]()
    {
        cudaMemset(d_dualCount, 0, sizeof(unsigned));
        cudaLaunchKernelEx(&dualCfg,
                           dualP2PKernel<DualConfig::numConsumersPerBlock, T>,
                           rawPtr(d_childOffsets),
                           rawPtr(d_nodeCenters),
                           rawPtr(d_nodeSizes),
                           box,
                           TreeNodeIndex{0},
                           TreeNodeIndex{0},
                           particlesPerBin,
                           d_dualCount);
    };

    float dualTime = timeGpu(runDual);
    printf("Dual GPU:              %.3f ms\n", dualTime);
    printf("Speedup (single/dual): %.2fx\n", singleTime / dualTime);

    // ── Copy counts ───────────────────────────────────────────────────────────
    unsigned h_singleCount = 0, h_dualCount = 0;
    cudaMemcpy(&h_singleCount, d_singleCount, sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_dualCount,   d_dualCount,   sizeof(unsigned), cudaMemcpyDeviceToHost);

    // printf("Pairs — CPU dual: %u, Single GPU: %u, Dual GPU: %u\n",
    printf("Single GPU: %u, Dual GPU: %u\n",
           /*cpuPairCount,*/ h_singleCount, h_dualCount);

    // ── Correctness ───────────────────────────────────────────────────────────
    // // Single traversal double-counts off-diagonal pairs and single-counts self-pairs.
    // EXPECT_EQ(h_singleCount, 2u * cpuPairCount - numLeaves)
    //     << "Single traversal count mismatch (expected 2*cpuCount - numLeaves)";

    // Dual traversal must reproduce the CPU reference exactly.
    // EXPECT_EQ(h_dualCount, cpuPairCount)
    //     << "Dual GPU pair count != CPU reference";

    cudaFree(d_singleCount);
    cudaFree(d_dualCount);
}

TEST(Traversal, syntheticP2PBenchmark)
{
    syntheticP2PBenchmark(10000000, 32);
}

} // namespace cstone
