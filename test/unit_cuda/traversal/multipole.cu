/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Multipole benchmark: GPU dual traversal vs. single traversal with theta-MAC
 *
 * @author Timo Schwab <tischwab@ethz.ch>
 *
 * Builds a tree from random Gaussian coordinates and evaluates a Barnes-Hut
 * theta-based multipole acceptance criterion (MAC). When two nodes are
 * sufficiently separated, an M2L interaction (simulated O(p^2) fmaf work) is
 * applied and all descendants are skipped. When nodes are too close, they are
 * refined until leaf-leaf P2P interactions (simulated O(n^2) fmaf work) remain.
 *
 * This demonstrates the core FMM advantage of dual traversal: early termination
 * of far-field node pairs via a single M2L, whereas single traversal must
 * independently discover this for every leaf.
 *
 * Two GPU methods are compared:
 *   1. Single: one GPU thread per leaf node calls singleTraversal. M2L work is
 *              done inside the criterion lambda (before returning false) since
 *              singleTraversal has no rejection callback.
 *   2. Dual:   cluster-based dualTraversalGPU with non-empty M2L and P2P.
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

using MultipoleTravConfig = TraversalConfig<1024, 224, 704, 512, 160, 96, 192, 640, 224, 32>;

// ── Single traversal GPU kernel ───────────────────────────────────────────────

template<class T>
__global__ void singleTraversalMultipoleKernel(
    const TreeNodeIndex* __restrict__ childOffsets,
    const TreeNodeIndex* __restrict__ parents,
    const TreeNodeIndex* __restrict__ leafToInternal,
    const Vec3<T>* __restrict__       nodeCenters,
    const Vec3<T>* __restrict__       nodeSizes,
    Box<T>                            box,
    T                                 invTheta,
    unsigned                          numLeaves,
    unsigned                          particlesPerBin,
    unsigned                          multipoleOrder,
    unsigned*                         p2pCount,
    unsigned*                         m2lCount)
{
    unsigned leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIdx >= numLeaves) { return; }

    TreeNodeIndex a   = leafToInternal[leafIdx];
    Vec3<T>  centerA  = nodeCenters[a];
    Vec3<T>  sizeA    = nodeSizes[a];

    // Criterion: returns true to continue descending, false to reject (M2L applied inline).
    // singleTraversal has no rejection callback, so M2L work must be done here before returning false.
    auto criterion = [a, centerA, sizeA, nodeCenters, nodeSizes, box, invTheta,
                      childOffsets, multipoleOrder, m2lCount]
        (TreeNodeIndex b) -> bool
    {
        Vec3<T> cB = nodeCenters[b];
        Vec3<T> sB = nodeSizes[b];

        Vec3<T> d = minDistance(centerA, sizeA, cB, sB, box);
        T dist2   = norm2(d);

        // l_max = max side length of either node (size stores half-sizes)
        T lA = T(2) * max(max(sizeA[0], sizeA[1]), sizeA[2]);
        T lB = T(2) * max(max(sB[0], sB[1]), sB[2]);
        T lMax = max(lA, lB);

        T threshold = lMax * invTheta;
        T threshold2 = threshold * threshold;

        if (dist2 < threshold2)
        {
            // Too close — continue refining
            return true;
        }

        // Far enough — apply M2L work inline before returning false
        // Only for internal nodes (leaves will be handled by endpoint action if they pass criterion)
        if (childOffsets[b] != 0)
        {
            volatile float acc = 0.f;
            for (unsigned pi = 0; pi < multipoleOrder; ++pi)
                for (unsigned pj = 0; pj < multipoleOrder; ++pj)
                    acc = __fmaf_rn(float(a + pi), float(b + pj), acc);
            atomicAdd(m2lCount, 1u);
        }
        return false;
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
__global__ void dualMultipoleKernel(
    const TreeNodeIndex* __restrict__ childOffsets,
    const Vec3<T>* __restrict__       nodeCenters,
    const Vec3<T>* __restrict__       nodeSizes,
    Box<T>                            box,
    T                                 invTheta,
    TreeNodeIndex                     rootA,
    TreeNodeIndex                     rootB,
    unsigned                          particlesPerBin,
    unsigned                          multipoleOrder,
    unsigned*                         p2pCount,
    unsigned*                         m2lCount,
    GlobalWorkQueue                   globalQueue,
    GlobalTraversalQueue              globalTraversalQueue,
    unsigned*                         numActiveProducers)
{
    auto criterion = [nodeCenters, nodeSizes, box, invTheta]
        __device__(TreeNodeIndex a, TreeNodeIndex b) -> bool
    {
        Vec3<T> cA = nodeCenters[a];
        Vec3<T> sA = nodeSizes[a];
        Vec3<T> cB = nodeCenters[b];
        Vec3<T> sB = nodeSizes[b];

        Vec3<T> d = minDistance(cA, sA, cB, sB, box);
        T dist2   = norm2(d);

        T lA = T(2) * max(max(sA[0], sA[1]), sA[2]);
        T lB = T(2) * max(max(sB[0], sB[1]), sB[2]);
        T lMax = max(lA, lB);

        T threshold = lMax * invTheta;
        return dist2 < threshold * threshold;
    };

    auto m2l = [m2lCount, multipoleOrder] __device__(TreeNodeIndex a, TreeNodeIndex b)
    {
        volatile float acc = 0.f;
        for (unsigned pi = 0; pi < multipoleOrder; ++pi)
            for (unsigned pj = 0; pj < multipoleOrder; ++pj)
                acc = __fmaf_rn(float(a + pi), float(b + pj), acc);
        atomicAdd(m2lCount, 1u);
    };

    auto p2p = [p2pCount, particlesPerBin] __device__(TreeNodeIndex a, TreeNodeIndex b)
    {
        volatile float acc = 0.f;
        for (unsigned pi = 0; pi < particlesPerBin; ++pi)
            for (unsigned pj = 0; pj < particlesPerBin; ++pj)
                acc = __fmaf_rn(float(a + pi), float(b + pj), acc);
        atomicAdd(p2pCount, 1u);
    };

    dualTraversalGPU<numWarps, MultipoleTravConfig>(
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

    static constexpr unsigned kBlocksPerCluster = 8;
    static constexpr unsigned kNumClusters      = 64;
    static constexpr unsigned kTotalBlocks      = kBlocksPerCluster * kNumClusters;

    static constexpr unsigned stackCap = MultipoleTravConfig::stackCap;
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

inline BenchStats computeStats(std::vector<float>& samples)
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

inline void printStats(const char* label, const BenchStats& s)
{
    printf("  %-22s  median=%.3f  mean=%.3f  stddev=%.3f  min=%.3f  max=%.3f ms\n",
           label, s.median, s.mean, s.stddev, s.minVal, s.maxVal);
}

// ── Benchmark body ────────────────────────────────────────────────────────────

void multipoleBenchmark(unsigned numParticles    = 2000000,
                        unsigned particlesPerBin = 32,
                        unsigned multipoleOrder  = 4,
                        double   theta           = 0.5,
                        unsigned bucketSize      = 16,
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
                         bucketSize, leaves, counts))
        ;

    Octree<KeyType> octree;
    octree.update(leaves.data(), nNodes(leaves));

    const TreeNodeIndex numTreeNodes = octree.numTreeNodes();
    const unsigned      numLeaves    = nNodes(leaves);

    T invTheta = T(1) / T(theta);

    printf("Multipole benchmark (theta=%.2f, p=%u, ppb=%u)\n", theta, multipoleOrder, particlesPerBin);
    printf("  particles=%u  leaves=%u  nodes=%d  bucket=%u\n",
           numParticles, numLeaves, numTreeNodes, bucketSize);
    printf("  P2P cost=%u  M2L cost=%u  ratio=%u:1\n",
           particlesPerBin * particlesPerBin, multipoleOrder * multipoleOrder,
           (particlesPerBin * particlesPerBin) / (multipoleOrder * multipoleOrder));
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

    unsigned* d_p2pCount;
    unsigned* d_m2lCount;
    cudaMalloc(&d_p2pCount, sizeof(unsigned));
    cudaMalloc(&d_m2lCount, sizeof(unsigned));

    // ── Single traversal lambda ──────────────────────────────────────────────
    const unsigned singleBlocks =
        (numLeaves + SingleConfig::numThreadsPerBlock - 1) / SingleConfig::numThreadsPerBlock;

    auto runSingle = [&]()
    {
        cudaMemset(d_p2pCount, 0, sizeof(unsigned));
        cudaMemset(d_m2lCount, 0, sizeof(unsigned));
        singleTraversalMultipoleKernel<T>
            <<<singleBlocks, SingleConfig::numThreadsPerBlock>>>(
                rawPtr(d_childOffsets),
                rawPtr(d_parents),
                rawPtr(d_leafToInternal),
                rawPtr(d_nodeCenters),
                rawPtr(d_nodeSizes),
                box,
                invTheta,
                numLeaves,
                particlesPerBin,
                multipoleOrder,
                d_p2pCount,
                d_m2lCount);
    };

    // ── Global work buffer ──
    constexpr unsigned gChunk = MultipoleTravConfig::chunkSize;
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
    constexpr unsigned tChunk = MultipoleTravConfig::travChunkSize;
    constexpr unsigned tSegs  = 2048;
    constexpr unsigned tCap   = tSegs * tChunk;
    TreeNodeIndex* d_tA;  cudaMalloc(&d_tA, tCap * sizeof(TreeNodeIndex));
    TreeNodeIndex* d_tB;  cudaMalloc(&d_tB, tCap * sizeof(TreeNodeIndex));
    unsigned* d_twHead;   cudaMalloc(&d_twHead, sizeof(unsigned));
    unsigned* d_trHead;   cudaMalloc(&d_trHead, sizeof(unsigned));
    unsigned* d_tsegR;    cudaMalloc(&d_tsegR, tSegs * sizeof(unsigned));
    GlobalTraversalQueue tq{d_tA, d_tB, d_twHead, d_trHead, d_tsegR, tSegs, tChunk};

    unsigned dualTotalBlocks = DualConfig::kTotalBlocks;

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
        cudaMemset(d_p2pCount, 0, sizeof(unsigned));
        cudaMemset(d_m2lCount, 0, sizeof(unsigned));
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
                           dualMultipoleKernel<DualConfig::numWarps, T>,
                           rawPtr(d_childOffsets),
                           rawPtr(d_nodeCenters),
                           rawPtr(d_nodeSizes),
                           box,
                           invTheta,
                           TreeNodeIndex{0},
                           TreeNodeIndex{0},
                           particlesPerBin,
                           multipoleOrder,
                           d_p2pCount,
                           d_m2lCount,
                           gq,
                           tq,
                           d_nProd);
    };

    // ── Correctness check (one run of each) ──────────────────────────────────
    runSingle();
    cudaDeviceSynchronize();
    unsigned h_singleP2P = 0, h_singleM2L = 0;
    cudaMemcpy(&h_singleP2P, d_p2pCount, sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_singleM2L, d_m2lCount, sizeof(unsigned), cudaMemcpyDeviceToHost);

    runDual();
    cudaDeviceSynchronize();
    unsigned h_dualP2P = 0, h_dualM2L = 0;
    cudaMemcpy(&h_dualP2P, d_p2pCount, sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_dualM2L, d_m2lCount, sizeof(unsigned), cudaMemcpyDeviceToHost);

    printf("  P2P pairs:  single=%u  dual=%u\n", h_singleP2P, h_dualP2P);
    printf("  M2L pairs:  single=%u  dual=%u\n", h_singleM2L, h_dualM2L);
    if (h_dualM2L > 0)
        printf("  M2L reduction ratio: %.1fx (single/dual)\n\n", (float)h_singleM2L / (float)h_dualM2L);
    else
        printf("  M2L reduction ratio: N/A (dual M2L count is 0)\n\n");

    // ── Warmup ───────────────────────────────────────────────────────────────
    printf("  Warming up (%u runs each)...\n", numWarmup);
    for (unsigned i = 0; i < numWarmup; ++i) { timeGpu(runSingle); }
    for (unsigned i = 0; i < numWarmup; ++i) { timeGpu(runDual); }

    // ── Timed runs ───────────────────────────────────────────────────────────
    printf("  Timing (%u runs each)...\n\n", numRuns);

    std::vector<float> singleTimes(numRuns);
    std::vector<float> dualTimes(numRuns);

    for (unsigned i = 0; i < numRuns; ++i) { singleTimes[i] = timeGpu(runSingle); }
    for (unsigned i = 0; i < numRuns; ++i) { dualTimes[i]   = timeGpu(runDual); }

    // ── Report ───────────────────────────────────────────────────────────────
    BenchStats singleStats = computeStats(singleTimes);
    BenchStats dualStats   = computeStats(dualTimes);

    printStats("Single traversal:", singleStats);
    printStats("Dual traversal:",   dualStats);

    printf("\n  Speedup (median): %.2fx\n", singleStats.median / dualStats.median);
    printf("  Speedup (mean):   %.2fx\n",   singleStats.mean / dualStats.mean);

    // ── Cleanup ──────────────────────────────────────────────────────────────
    cudaFree(d_p2pCount);
    cudaFree(d_m2lCount);
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

TEST(Traversal, multipoleBenchmark)
{
    multipoleBenchmark(2000000, 32, 4, 0.5, 16, 10, 50);
}

} // namespace cstone
