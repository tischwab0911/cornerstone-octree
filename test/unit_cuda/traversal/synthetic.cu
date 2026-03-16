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
 * The p2p action spins over a (particlesPerBin x particlesPerBin) fmaf loop to
 * simulate the actual per-particle work inside a leaf-leaf interaction.
 *
 * Benchmark protocol: 20 warmup runs (discarded), then 100 timed runs.
 * Reports median, mean, stddev, min, max for each method.
 */

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
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

using DefaultTravConfig = TraversalConfig<1024, 224, 704, 512, 160, 96, 192, 640, 224, 32>;

// ── Single traversal GPU kernel ───────────────────────────────────────────────

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

    auto criterion = [centerA, sizeA, nodeCenters, nodeSizes, box]
        (TreeNodeIndex b) -> bool
    {
        Vec3<T> d = minDistance(centerA, sizeA, nodeCenters[b], nodeSizes[b], box);
        return norm2(d) == T(0);
    };

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

template<int numWarps, class T>
__global__ void dualP2PKernel(
    const TreeNodeIndex* __restrict__ childOffsets,
    const Vec3<T>* __restrict__       nodeCenters,
    const Vec3<T>* __restrict__       nodeSizes,
    Box<T>                            box,
    TreeNodeIndex                     rootA,
    TreeNodeIndex                     rootB,
    unsigned                          particlesPerBin,
    unsigned*                         p2pCount,
    GlobalWorkQueue                   globalQueue,
    GlobalTraversalQueue              globalTraversalQueue,
    unsigned*                         numActiveProducers)
{
    auto criterion = [nodeCenters, nodeSizes, box] __device__(TreeNodeIndex a, TreeNodeIndex b) -> bool
    {
        Vec3<T> d = minDistance(nodeCenters[a], nodeSizes[a], nodeCenters[b], nodeSizes[b], box);
        return norm2(d) == T(0);
    };

    auto m2l = [] __device__(TreeNodeIndex, TreeNodeIndex) {};

    auto p2p = [p2pCount, particlesPerBin] __device__(TreeNodeIndex a, TreeNodeIndex b)
    {
        volatile float acc = 0.f;
        for (unsigned pi = 0; pi < particlesPerBin; ++pi)
            for (unsigned pj = 0; pj < particlesPerBin; ++pj)
                acc = __fmaf_rn(float(a + pi), float(b + pj), acc);
        atomicAdd(p2pCount, 1u);
    };

    dualTraversalGPU<numWarps, DefaultTravConfig>(
        childOffsets, rootA, rootB,
        globalQueue, globalTraversalQueue, numActiveProducers,
        criterion, m2l, p2p);
}

// ── Launch configurations ─────────────────────────────────────────────────────

struct SingleConfig
{
    static constexpr unsigned numThreadsPerBlock = 256;
};

struct DualConfig
{
    static constexpr unsigned numWarps = 7;
    static constexpr unsigned numThreadsPerBlock = numWarps * GpuConfig::warpSize;
    // static_assert(numThreadsPerBlock >= 64 && numThreadsPerBlock <= 512);

    static constexpr unsigned kBlocksPerCluster = 8;
    static constexpr unsigned kNumClusters      = 64;
    static constexpr unsigned kTotalBlocks      = kBlocksPerCluster * kNumClusters;

    //! @brief per-block shared memory capacity (derived from TraversalConfig)
    static constexpr unsigned stackCap = DefaultTravConfig::stackCap;
};

// ── Statistics helpers ────────────────────────────────────────────────────────

struct BenchStats
{
    float median;
    float mean;
    float stddev;
    float minVal;
    float maxVal;
};

BenchStats computeStats(std::vector<float>& samples)
{
    std::sort(samples.begin(), samples.end());

    size_t n = samples.size();
    float med = (n % 2 == 1)
        ? samples[n / 2]
        : 0.5f * (samples[n / 2 - 1] + samples[n / 2]);

    float sum = std::accumulate(samples.begin(), samples.end(), 0.0f);
    float avg = sum / (float)n;

    float sqSum = 0.0f;
    for (float s : samples)
        sqSum += (s - avg) * (s - avg);
    float sd = std::sqrt(sqSum / (float)n);

    return {med, avg, sd, samples.front(), samples.back()};
}

void printStats(const char* label, const BenchStats& s)
{
    printf("  %-22s  median=%.3f  mean=%.3f  stddev=%.3f  min=%.3f  max=%.3f ms\n",
           label, s.median, s.mean, s.stddev, s.minVal, s.maxVal);
}

// ── Benchmark body ────────────────────────────────────────────────────────────

void syntheticP2PBenchmark(unsigned numParticles    = 1000000,
                            unsigned particlesPerBin = 32,
                            unsigned numWarmup       = 10,
                            unsigned numRuns         = 50)
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

    printf("Synthetic P2P benchmark\n");
    printf("  particles=%u  leaves=%u  nodes=%d  bin=%u\n",
           numParticles, numLeaves, numTreeNodes, particlesPerBin);
    printf("  warmup=%u  runs=%u\n\n", numWarmup, numRuns);

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

    // ── leafToInternal mapping ─────────────────────────────────────────────────
    std::vector<TreeNodeIndex> h_leafToInternal(numLeaves);
    for (unsigned i = 0; i < numLeaves; ++i)
        h_leafToInternal[i] = octree.toInternal(i);

    // ── Upload arrays to GPU ──────────────────────────────────────────────────
    auto co  = octree.childOffsets();
    auto par = octree.parents();

    DeviceVector<TreeNodeIndex> d_childOffsets(co.data(), co.data() + co.size());
    DeviceVector<TreeNodeIndex> d_parents(par.data(), par.data() + par.size());
    DeviceVector<TreeNodeIndex> d_leafToInternal(h_leafToInternal);
    DeviceVector<Vec3<T>>       d_nodeCenters(h_nodeCenters);
    DeviceVector<Vec3<T>>       d_nodeSizes(h_nodeSizes);

    unsigned* d_count;
    cudaMalloc(&d_count, sizeof(unsigned));

    // ── Single traversal lambda ──────────────────────────────────────────────
    const unsigned singleBlocks =
        (numLeaves + SingleConfig::numThreadsPerBlock - 1) / SingleConfig::numThreadsPerBlock;

    auto runSingle = [&]()
    {
        cudaMemset(d_count, 0, sizeof(unsigned));
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
                d_count);
    };

    // ── Global work buffer ──
    constexpr unsigned gChunk = DefaultTravConfig::chunkSize;
    constexpr unsigned gSegs  = 2048;
    constexpr unsigned gCap   = gSegs * gChunk;
    TreeNodeIndex* d_gA; cudaMalloc(&d_gA, gCap * sizeof(TreeNodeIndex));
    TreeNodeIndex* d_gB; cudaMalloc(&d_gB, gCap * sizeof(TreeNodeIndex));
    int* d_gIsP2P;       cudaMalloc(&d_gIsP2P, gCap * sizeof(int));
    unsigned* d_wHead;   cudaMalloc(&d_wHead, sizeof(unsigned));
    unsigned* d_rHead;   cudaMalloc(&d_rHead, sizeof(unsigned));
    unsigned* d_segCount; cudaMalloc(&d_segCount, gSegs * sizeof(unsigned));
    unsigned* d_segR;    cudaMalloc(&d_segR, gSegs * sizeof(unsigned));
    unsigned* d_nProd;   cudaMalloc(&d_nProd, sizeof(unsigned));
    GlobalWorkQueue gq{d_gA, d_gB, d_gIsP2P, d_wHead, d_rHead, d_segCount, d_segR, gSegs, gChunk};

    // Global traversal queue
    constexpr unsigned tChunk = DefaultTravConfig::travChunkSize;
    constexpr unsigned tSegs  = 2048;
    constexpr unsigned tCap   = tSegs * tChunk;
    TreeNodeIndex* d_tA;  cudaMalloc(&d_tA, tCap * sizeof(TreeNodeIndex));
    TreeNodeIndex* d_tB;  cudaMalloc(&d_tB, tCap * sizeof(TreeNodeIndex));
    unsigned* d_twHead;   cudaMalloc(&d_twHead, sizeof(unsigned));
    unsigned* d_trHead;   cudaMalloc(&d_trHead, sizeof(unsigned));
    unsigned* d_tsegR;    cudaMalloc(&d_tsegR, tSegs * sizeof(unsigned));
    GlobalTraversalQueue tq{d_tA, d_tB, d_twHead, d_trHead, d_tsegR, tSegs, tChunk};

    // unsigned maxBlocks = maxConcurrentBlocks(
    //     dualP2PKernel<DualConfig::numWarps, T>,
    //     DualConfig::numThreadsPerBlock, DualConfig::kBlocksPerCluster);
    // unsigned dualTotalBlocks = std::min(DualConfig::kTotalBlocks, maxBlocks);
    unsigned dualTotalBlocks = DualConfig::kTotalBlocks;
    // printf("Max concurrent blocks for dual traversal: %u (configured %u)\n", maxBlocks, DualConfig::kTotalBlocks);

    // ── Dual traversal lambda ────────────────────────────────────────────────
    cudaLaunchConfig_t dualCfg{};
    dualCfg.gridDim  = {dualTotalBlocks, 1, 1};
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
        cudaMemset(d_count, 0, sizeof(unsigned));
        cudaMemset(d_wHead, 0, sizeof(unsigned));
        cudaMemset(d_rHead, 0, sizeof(unsigned));
        cudaMemset(d_segCount, 0, gSegs * sizeof(unsigned));
        cudaMemset(d_segR, 0, gSegs * sizeof(unsigned));
        cudaMemset(d_twHead, 0, sizeof(unsigned));
        cudaMemset(d_trHead, 0, sizeof(unsigned));
        cudaMemset(d_tsegR, 0, tSegs * sizeof(unsigned));
        unsigned producerWarpsTotal = dualTotalBlocks * 2u;
        cudaMemcpy(d_nProd, &producerWarpsTotal, sizeof(unsigned), cudaMemcpyHostToDevice);
        cudaLaunchKernelEx(&dualCfg,
                           dualP2PKernel<DualConfig::numWarps, T>,
                           rawPtr(d_childOffsets),
                           rawPtr(d_nodeCenters),
                           rawPtr(d_nodeSizes),
                           box,
                           TreeNodeIndex{0},
                           TreeNodeIndex{0},
                           particlesPerBin,
                           d_count,
                           gq,
                           tq,
                           d_nProd);
    };

    // ── Correctness check (one run of each) ──────────────────────────────────
    runSingle();
    cudaDeviceSynchronize();
    unsigned h_singleCount = 0;
    cudaMemcpy(&h_singleCount, d_count, sizeof(unsigned), cudaMemcpyDeviceToHost);

    runDual();
    cudaDeviceSynchronize();
    unsigned h_dualCount = 0;
    cudaMemcpy(&h_dualCount, d_count, sizeof(unsigned), cudaMemcpyDeviceToHost);

    printf("  p2p pairs: single=%u  dual=%u\n\n", h_singleCount, h_dualCount);

    // ── Warmup ───────────────────────────────────────────────────────────────
    printf("  Warming up (%u runs each)...\n", numWarmup);
    for (unsigned i = 0; i < numWarmup; ++i) { timeGpu(runSingle); printf("%d\n",i);}
    for (unsigned i = 0; i < numWarmup; ++i) { timeGpu(runDual); printf("%d\n",i);}

    // ── Timed runs ───────────────────────────────────────────────────────────
    printf("  Timing (%u runs each)...\n\n", numRuns);

    std::vector<float> singleTimes(numRuns);
    std::vector<float> dualTimes(numRuns);

    for (unsigned i = 0; i < numRuns; ++i) { singleTimes[i] = timeGpu(runSingle); }
    for (unsigned i = 0; i < numRuns; ++i) { dualTimes[i]   = timeGpu(runDual); printf("%d\n",i);}

    // ── Report ───────────────────────────────────────────────────────────────
    BenchStats singleStats = computeStats(singleTimes);
    BenchStats dualStats   = computeStats(dualTimes);

    printStats("Single traversal:", singleStats);
    printStats("Dual traversal:",   dualStats);

    printf("\n  Speedup (median): %.2fx\n", singleStats.median / dualStats.median);
    printf("  Speedup (mean):   %.2fx\n",   singleStats.mean / dualStats.mean);

    cudaFree(d_count);
    cudaFree(d_gA);
    cudaFree(d_gB);
    cudaFree(d_gIsP2P);
    cudaFree(d_wHead);
    cudaFree(d_rHead);
    cudaFree(d_segCount);
    cudaFree(d_segR);
    cudaFree(d_nProd);
    cudaFree(d_tA);
    cudaFree(d_tB);
    cudaFree(d_twHead);
    cudaFree(d_trHead);
    cudaFree(d_tsegR);
}

TEST(Traversal, syntheticP2PBenchmark)
{
    syntheticP2PBenchmark(5000000, 16, 10, 50);
}

} // namespace cstone
