/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Parameter tuning benchmark for GPU dual traversal
 *
 * @author Timo Schwab <tischwab@ethz.ch>
 *
 * Sweeps over numWarps and ALL TraversalConfig parameters
 * (stackCap, interaction push/pop thresholds, traversal push/pop thresholds)
 * to find optimal settings for the dual traversal GPU kernel.
 *
 * blocksPerCluster is fixed at 8.
 *
 * Uses the same workload as synthetic.cu: random Gaussian coordinates,
 * minDistance==0 criterion, fmaf P2P work loop.
 *
 * Benchmark protocol: 5 warmup runs (discarded), then 20 timed runs.
 * Reports all parameters and timing for every tested configuration.
 *
 * TO CUSTOMIZE: Edit the section marked "USER-EDITABLE TUNING RANGES" below.
 */

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <utility>

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

// ════════════════════════════════════════════════════════════════════════════════
//  USER-EDITABLE TUNING RANGES
//
//  Modify the values below to control which parameter combinations are tested.
//  After editing, rebuild and re-run the benchmark.
// ════════════════════════════════════════════════════════════════════════════════

//  numWarps values to benchmark (consumer warps per block = numWarps - 1).
//  Edit the values inside the angle brackets:
using TuneWarpList = std::integer_sequence<int, 4, 5, 6>;

//  blocksPerCluster — fixed for all configurations (not tuned):
static constexpr unsigned kBpc = 8;

//  TraversalConfig parameter set.
//  Each field maps 1:1 to the template parameters of TraversalConfig<...>.
struct TuneParams
{
    unsigned stackCap;          //  shared-memory buffer capacity (node-pair slots)
    // ── interaction buffer thresholds ──
    unsigned chunkSize;         //  items per global interaction push/pop
    unsigned forcePush;         //  local count above this -> must push to global
    unsigned attemptPush;       //  local count above this -> try to push to global
    unsigned attemptPop;        //  local count below this -> try to pop from global
    unsigned forcePop;          //  local count below this -> only pop locally
    // ── traversal stack thresholds ──
    unsigned travChunkSize;     //  items per global traversal push/pop
    unsigned travForcePush;     //  stack depth above this -> must push to global
    unsigned travAttemptPush;   //  stack depth above this -> try to push to global
    unsigned travAttemptPop;    //  stack depth below this -> try to pop from global
};

//  Helper: default TraversalConfig ratios for a given stackCap.
static constexpr TuneParams defaults(unsigned sc)
{
    return {sc,
            128,                          // chunkSize
            640,                      // forcePush
            448,                     // attemptPush
            192,                     // attemptPop
            64,                     // forcePop
            256,                          // travChunkSize
            sc - 8 * GpuConfig::warpSize,    // travForcePush
            256,                     // travAttemptPush
            32};                    // travAttemptPop
}

//  Modifier helpers — create a variant from an existing config:
static constexpr TuneParams withChunkSize      (TuneParams p, unsigned v) { p.chunkSize       = v; return p; }
static constexpr TuneParams withForcePush      (TuneParams p, unsigned v) { p.forcePush       = v; return p; }
static constexpr TuneParams withAttemptPush    (TuneParams p, unsigned v) { p.attemptPush     = v; return p; }
static constexpr TuneParams withAttemptPop     (TuneParams p, unsigned v) { p.attemptPop      = v; return p; }
static constexpr TuneParams withForcePop       (TuneParams p, unsigned v) { p.forcePop        = v; return p; }
static constexpr TuneParams withTravChunkSize  (TuneParams p, unsigned v) { p.travChunkSize   = v; return p; }
static constexpr TuneParams withTravForcePush  (TuneParams p, unsigned v) { p.travForcePush   = v; return p; }
static constexpr TuneParams withTravAttemptPush(TuneParams p, unsigned v) { p.travAttemptPush = v; return p; }
static constexpr TuneParams withTravAttemptPop (TuneParams p, unsigned v) { p.travAttemptPop  = v; return p; }

//  ┌──────────────────────────────────────────────────────────────────────────┐
//  │  CONFIGURATION TABLE                                                     │
//  │  Add, remove, or modify entries freely.  Each entry is one               │
//  │  TraversalConfig instantiation that will be tested with every numWarps.  │
//  └──────────────────────────────────────────────────────────────────────────┘
static constexpr TuneParams tuneConfigs[] = {

    // ── Baseline: vary stackCap with default ratios ──
    // defaults(512),
    // defaults(768),
    defaults(1024),
    defaults(1280),
    // defaults(1536),

    // ── Interaction: vary chunkSize (base: defaults(1024)) ──
    withChunkSize(defaults(1024), 64),
    withChunkSize(defaults(1024), 128),
    withChunkSize(defaults(1024), 192),
    // withChunkSize(defaults(1024), 256),

    // ── Interaction: vary forcePush ──
    withForcePush(defaults(1024), 480),
    withForcePush(defaults(1024), 768),
    withForcePush(defaults(1024), 896),

    // ── Interaction: vary attemptPush ──
    withAttemptPush(defaults(1024), 192),
    // withAttemptPush(defaults(1024), 448),
    withAttemptPush(defaults(1024), 576),

    // ── Interaction: vary attemptPop ──
    withAttemptPop(defaults(1024), 64),
    withAttemptPop(defaults(1024), 128),
    withAttemptPop(defaults(1024), 256),
    withAttemptPop(defaults(1024), 320),

    // ── Interaction: vary forcePop ──
    withForcePop(defaults(1024), 32),
    withForcePop(defaults(1024), 96),
    withForcePop(defaults(1024), 128),
    // withForcePop(defaults(1024), 192),

    // ── Traversal: vary travChunkSize ──
    // withTravChunkSize(defaults(1024), 64),
    // withTravChunkSize(defaults(1024), 192),
    // withTravChunkSize(defaults(1024), 320),

    // ── Traversal: vary travForcePush ──
    withTravForcePush(defaults(1024), 640),
    withTravForcePush(defaults(1024), 896),

    // ── Traversal: vary travAttemptPush ──
    // withTravAttemptPush(defaults(1024), 128),
    // withTravAttemptPush(defaults(1024), 320),
    // withTravAttemptPush(defaults(1024), 512),

    // ── Traversal: vary travAttemptPop ──
    // withTravAttemptPop(defaults(1024), 32),
    withTravAttemptPop(defaults(1024), 50),
    // withTravAttemptPop(defaults(1024), 128),
    // withTravAttemptPop(defaults(1024), 192),
};

static constexpr size_t numTuneConfigs = sizeof(tuneConfigs) / sizeof(tuneConfigs[0]);

// ════════════════════════════════════════════════════════════════════════════════
//  END OF USER-EDITABLE SECTION
// ════════════════════════════════════════════════════════════════════════════════

// ── Statistics ────────────────────────────────────────────────────────────────

struct TuneStats
{
    float median, mean, stddev, minVal, maxVal;
};

static TuneStats computeStats(std::vector<float>& v)
{
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    float med = (n % 2) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
    float sum = std::accumulate(v.begin(), v.end(), 0.f);
    float avg = sum / float(n);
    float sq  = 0.f;
    for (float s : v) sq += (s - avg) * (s - avg);
    return {med, avg, std::sqrt(sq / float(n)), v.front(), v.back()};
}

// ── Result record ─────────────────────────────────────────────────────────────

struct TuneResult
{
    int        numWarps;
    TuneParams params;
    int        totalBlocks;
    bool       valid;
    unsigned   p2pCount;
    unsigned   iactWHead, iactRHead;   // interaction queue write/read heads
    unsigned   travWHead, travRHead;   // traversal queue write/read heads
    TuneStats  stats;
};

// ── Compile-time mapping: TuneParams[I] -> TraversalConfig<...> ──────────────

template<size_t I>
using TuneTravConfig = TraversalConfig<
    tuneConfigs[I].stackCap,
    tuneConfigs[I].chunkSize,
    tuneConfigs[I].forcePush,
    tuneConfigs[I].attemptPush,
    tuneConfigs[I].attemptPop,
    tuneConfigs[I].forcePop,
    tuneConfigs[I].travChunkSize,
    tuneConfigs[I].travForcePush,
    tuneConfigs[I].travAttemptPush,
    tuneConfigs[I].travAttemptPop>;

// ── Single traversal P2P kernel (baseline) ───────────────────────────────────

template<class T>
__global__ void tuneSingleP2PKernel(
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
    if (leafIdx >= numLeaves) return;

    TreeNodeIndex a  = leafToInternal[leafIdx];
    Vec3<T> centerA  = nodeCenters[a];
    Vec3<T> sizeA    = nodeSizes[a];

    auto criterion = [centerA, sizeA, nodeCenters, nodeSizes, box]
        (TreeNodeIndex b) -> bool
    {
        Vec3<T> d = minDistance(centerA, sizeA, nodeCenters[b], nodeSizes[b], box);
        return norm2(d) == T(0);
    };

    auto action = [p2pCount, a, particlesPerBin](TreeNodeIndex b)
    {
        volatile float acc = 0.f;
        for (unsigned pi = 0; pi < particlesPerBin; ++pi)
            for (unsigned pj = 0; pj < particlesPerBin; ++pj)
                acc = __fmaf_rn(float(a + pi), float(b + pj), acc);
        atomicAdd(p2pCount, 1u);
    };

    singleTraversal(childOffsets, parents, criterion, action);
}

// ── Dual P2P kernel ──────────────────────────────────────────────────────────

template<int numWarps, class TravConfig, class T>
__global__ void tuneDualP2PKernel(
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
    auto criterion = [nodeCenters, nodeSizes, box] __device__
                     (TreeNodeIndex a, TreeNodeIndex b) -> bool
    {
        Vec3<T> d = minDistance(nodeCenters[a], nodeSizes[a],
                                nodeCenters[b], nodeSizes[b], box);
        return norm2(d) == T(0);
    };

    auto m2l = [] __device__(TreeNodeIndex, TreeNodeIndex) {};

    auto p2p = [p2pCount, particlesPerBin] __device__
               (TreeNodeIndex a, TreeNodeIndex b)
    {
        volatile float acc = 0.f;
        for (unsigned pi = 0; pi < particlesPerBin; ++pi)
            for (unsigned pj = 0; pj < particlesPerBin; ++pj)
                acc = __fmaf_rn(float(a + pi), float(b + pj), acc);
        atomicAdd(p2pCount, 1u);
    };

    dualTraversalGPU<numWarps, TravConfig>(
        childOffsets, rootA, rootB,
        globalQueue, globalTraversalQueue, numActiveProducers,
        criterion, m2l, p2p);
}

// ── Benchmark one (numWarps, TravConfig) combination ──────────────────────────

template<int numWarps, class TravConfig, class T>
TuneResult benchOne(
    const TuneParams&            params,
    DeviceVector<TreeNodeIndex>& d_co,
    DeviceVector<Vec3<T>>&       d_cen,
    DeviceVector<Vec3<T>>&       d_sz,
    Box<T>                       box,
    unsigned                     ppb,
    unsigned*                    d_count,
    unsigned                     nWarm,
    unsigned                     nRuns)
{
    constexpr unsigned tpb = numWarps * GpuConfig::warpSize;
    TuneResult res{numWarps, params, 0, false, 0, {}};

    unsigned maxBlks = maxConcurrentBlocks(
        tuneDualP2PKernel<numWarps, TravConfig, T>, tpb, kBpc);
    if (maxBlks > 100000u) return res;

    unsigned total = std::min(kBpc * 64u, maxBlks);
    total = (total / kBpc) * kBpc;
    if (total == 0) return res;
    res.totalBlocks = int(total);

    // ── Allocate global interaction queue ──
    constexpr unsigned gChunk = TravConfig::chunkSize;
    constexpr unsigned gSegs  = 1024;
    constexpr unsigned gCap   = gSegs * gChunk;

    TreeNodeIndex *d_gA, *d_gB;
    int *d_gIsP2P;
    unsigned *d_wH, *d_rH, *d_segCount, *d_sR, *d_nP;
    cudaMalloc(&d_gA,   gCap * sizeof(TreeNodeIndex));
    cudaMalloc(&d_gB,   gCap * sizeof(TreeNodeIndex));
    cudaMalloc(&d_gIsP2P, gCap * sizeof(int));
    cudaMalloc(&d_wH,   sizeof(unsigned));
    cudaMalloc(&d_rH,   sizeof(unsigned));
    cudaMalloc(&d_segCount, gSegs * sizeof(unsigned));
    cudaMalloc(&d_sR,   gSegs * sizeof(unsigned));
    cudaMalloc(&d_nP,   sizeof(unsigned));
    GlobalWorkQueue gq{d_gA, d_gB, d_gIsP2P, d_wH, d_rH, d_segCount, d_sR, gSegs};

    // ── Allocate global traversal queue ──
    constexpr unsigned tChunk = TravConfig::travChunkSize;
    constexpr unsigned tSegs  = 1024;
    constexpr unsigned tCap   = tSegs * tChunk;

    TreeNodeIndex *d_tA, *d_tB;
    unsigned *d_twH, *d_trH, *d_tsR;
    cudaMalloc(&d_tA,  tCap * sizeof(TreeNodeIndex));
    cudaMalloc(&d_tB,  tCap * sizeof(TreeNodeIndex));
    cudaMalloc(&d_twH, sizeof(unsigned));
    cudaMalloc(&d_trH, sizeof(unsigned));
    cudaMalloc(&d_tsR, tSegs * sizeof(unsigned));
    GlobalTraversalQueue tq{d_tA, d_tB, d_twH, d_trH, d_tsR, tSegs};

    // ── Launch config ──
    cudaLaunchConfig_t cfg{};
    cfg.gridDim  = {total, 1, 1};
    cfg.blockDim = {tpb, 1, 1};

    cudaLaunchAttribute attr{};
    attr.id             = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {kBpc, 1, 1};
    cfg.attrs    = &attr;
    cfg.numAttrs = 1;

    auto run = [&]()
    {
        cudaMemset(d_count, 0, sizeof(unsigned));
        cudaMemset(d_wH,  0, sizeof(unsigned));
        cudaMemset(d_rH,  0, sizeof(unsigned));
        cudaMemset(d_segCount, 0, gSegs * sizeof(unsigned));
        cudaMemset(d_sR,  0, gSegs * sizeof(unsigned));
        cudaMemset(d_twH, 0, sizeof(unsigned));
        cudaMemset(d_trH, 0, sizeof(unsigned));
        cudaMemset(d_tsR, 0, tSegs * sizeof(unsigned));
        cudaMemcpy(d_nP, &total, sizeof(unsigned), cudaMemcpyHostToDevice);
        cudaLaunchKernelEx(&cfg,
            tuneDualP2PKernel<numWarps, TravConfig, T>,
            rawPtr(d_co), rawPtr(d_cen), rawPtr(d_sz),
            box, TreeNodeIndex{0}, TreeNodeIndex{0},
            ppb, d_count, gq, tq, d_nP);
        printf(".");
    };

    // ── Validation run ──
    run();
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess)
    {
        cudaMemcpy(&res.p2pCount, d_count, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&res.iactWHead, d_wH, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&res.iactRHead, d_rH, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&res.travWHead, d_twH, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&res.travRHead, d_trH, sizeof(unsigned), cudaMemcpyDeviceToHost);

        for (unsigned i = 0; i < nWarm; ++i) timeGpu(run);

        std::vector<float> times(nRuns);
        for (unsigned i = 0; i < nRuns; ++i) times[i] = timeGpu(run);
        res.stats = computeStats(times);
        res.valid = true;
    }
    else
    {
        cudaGetLastError();
    }

    cudaFree(d_gA); cudaFree(d_gB); cudaFree(d_gIsP2P);
    cudaFree(d_wH); cudaFree(d_rH); cudaFree(d_segCount); cudaFree(d_sR); cudaFree(d_nP);
    cudaFree(d_tA); cudaFree(d_tB);
    cudaFree(d_twH); cudaFree(d_trH); cudaFree(d_tsR);
    return res;
}

// ── Config dispatch: benchmark + record one config for one numWarps ──────────

template<int NW, size_t CI, class T>
void benchAndRecord(
    std::vector<TuneResult>& out,
    DeviceVector<TreeNodeIndex>& d_co,
    DeviceVector<Vec3<T>>& d_cen,
    DeviceVector<Vec3<T>>& d_sz,
    Box<T> box, unsigned ppb, unsigned* d_cnt,
    unsigned nWarm, unsigned nRuns)
{
    constexpr auto c = tuneConfigs[CI];
    printf("  [nW=%d] Config %2zu: SC=%-5u iCS=%-3u iFP=%-4u iAP=%-4u iAPo=%-3u iFPo=%-3u "
           "tCS=%-3u tFP=%-4u tAP=%-4u tAPo=%-3u ... ",
           NW, CI, c.stackCap,
           c.chunkSize, c.forcePush, c.attemptPush, c.attemptPop, c.forcePop,
           c.travChunkSize, c.travForcePush, c.travAttemptPush, c.travAttemptPop);
    fflush(stdout);

    auto r = benchOne<NW, TuneTravConfig<CI>, T>(
        c, d_co, d_cen, d_sz, box, ppb, d_cnt, nWarm, nRuns);

    if (r.valid)
        printf("median=%.3f ms\n", r.stats.median);
    else
        printf("SKIPPED\n");

    out.push_back(r);
}

// ── Iterate over all configs for one numWarps value ──────────────────────────

template<int NW, class T, size_t... CIs>
void benchAllConfigs(
    std::index_sequence<CIs...>,
    std::vector<TuneResult>& out,
    DeviceVector<TreeNodeIndex>& d_co,
    DeviceVector<Vec3<T>>& d_cen,
    DeviceVector<Vec3<T>>& d_sz,
    Box<T> box, unsigned ppb, unsigned* d_cnt,
    unsigned nWarm, unsigned nRuns)
{
    (benchAndRecord<NW, CIs, T>(out, d_co, d_cen, d_sz, box, ppb, d_cnt, nWarm, nRuns), ...);
}

// ── Iterate over all numWarps values (recursive integer_sequence dispatch) ───

template<class T, int First, int... Rest>
void dispatchWarps(
    std::integer_sequence<int, First, Rest...>,
    std::vector<TuneResult>& out,
    DeviceVector<TreeNodeIndex>& d_co,
    DeviceVector<Vec3<T>>& d_cen,
    DeviceVector<Vec3<T>>& d_sz,
    Box<T> box, unsigned ppb, unsigned* d_cnt,
    unsigned nWarm, unsigned nRuns)
{
    printf("\n── numWarps = %d ──────────────────────────────────────────────────────"
           "──────────────────────────────────────────────\n", First);
    benchAllConfigs<First, T>(
        std::make_index_sequence<numTuneConfigs>{},
        out, d_co, d_cen, d_sz, box, ppb, d_cnt, nWarm, nRuns);

    if constexpr (sizeof...(Rest) > 0)
    {
        dispatchWarps<T>(
            std::integer_sequence<int, Rest...>{},
            out, d_co, d_cen, d_sz, box, ppb, d_cnt, nWarm, nRuns);
    }
}

// ── Main benchmark ────────────────────────────────────────────────────────────

void tuneP2PBenchmark(unsigned numParticles    = 5000000,
                       unsigned particlesPerBin = 16,
                       unsigned numWarmup       = 5,
                       unsigned numRuns         = 20)
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

    printf("════════════════════════════════════════════════════════════════════════"
           "══════════════════════════════════════════════════════════════════════\n");
    printf("  Dual Traversal Parameter Tuning Benchmark\n");
    printf("════════════════════════════════════════════════════════════════════════"
           "══════════════════════════════════════════════════════════════════════\n");
    printf("  particles=%u  leaves=%u  nodes=%d  bin=%u\n",
           numParticles, numLeaves, numTreeNodes, particlesPerBin);
    printf("  warmup=%u  runs=%u  bpc=%u (fixed)\n", numWarmup, numRuns, kBpc);
    printf("  configs=%zu  numWarps values=%zu  total benchmarks=%zu\n",
           numTuneConfigs, TuneWarpList::size(), numTuneConfigs * TuneWarpList::size());
    printf("════════════════════════════════════════════════════════════════════════"
           "══════════════════════════════════════════════════════════════════════\n");

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

    // ── Upload to GPU ─────────────────────────────────────────────────────────
    auto co  = octree.childOffsets();
    auto par = octree.parents();

    DeviceVector<TreeNodeIndex> d_childOffsets(co.data(), co.data() + co.size());
    DeviceVector<TreeNodeIndex> d_parents(par.data(), par.data() + par.size());
    DeviceVector<TreeNodeIndex> d_leafToInternal(h_leafToInternal);
    DeviceVector<Vec3<T>>       d_nodeCenters(h_nodeCenters);
    DeviceVector<Vec3<T>>       d_nodeSizes(h_nodeSizes);

    unsigned* d_count;
    cudaMalloc(&d_count, sizeof(unsigned));

    // ── Single traversal baseline ─────────────────────────────────────────────
    constexpr unsigned kSingleTPB = 256;
    const unsigned singleBlocks = (numLeaves + kSingleTPB - 1) / kSingleTPB;

    auto runSingle = [&]()
    {
        cudaMemset(d_count, 0, sizeof(unsigned));
        tuneSingleP2PKernel<T><<<singleBlocks, kSingleTPB>>>(
            rawPtr(d_childOffsets), rawPtr(d_parents), rawPtr(d_leafToInternal),
            rawPtr(d_nodeCenters), rawPtr(d_nodeSizes),
            box, numLeaves, particlesPerBin, d_count);
    };

    printf("\nBenchmarking single traversal baseline...\n");
    runSingle();
    cudaDeviceSynchronize();
    unsigned singleP2P = 0;
    cudaMemcpy(&singleP2P, d_count, sizeof(unsigned), cudaMemcpyDeviceToHost);
    printf("  single p2p count = %u\n", singleP2P);

    for (unsigned i = 0; i < numWarmup; ++i) timeGpu(runSingle);
    std::vector<float> singleTimes(numRuns);
    for (unsigned i = 0; i < numRuns; ++i) singleTimes[i] = timeGpu(runSingle);
    TuneStats singleStats = computeStats(singleTimes);
    printf("  single traversal: median=%.3f  mean=%.3f  stddev=%.3f  min=%.3f  max=%.3f ms\n",
           singleStats.median, singleStats.mean, singleStats.stddev,
           singleStats.minVal, singleStats.maxVal);

    // ── Run all combinations ──────────────────────────────────────────────────
    std::vector<TuneResult> results;
    results.reserve(numTuneConfigs * TuneWarpList::size());

    printf("\nRunning benchmarks...\n");
    dispatchWarps<T>(TuneWarpList{}, results,
                     d_childOffsets, d_nodeCenters, d_nodeSizes,
                     box, particlesPerBin, d_count, numWarmup, numRuns);

    // ── Report ────────────────────────────────────────────────────────────────
    printf("\n");
    printf("════════════════════════════════════════════════════════════════════════"
           "══════════════════════════════════════════════════════════════════════"
           "════════════════════════════════════\n");
    printf("  Single traversal baseline: median=%.3f ms\n\n", singleStats.median);
    printf("  nW |   SC  | iCS  iFP  iAP iAPo iFPo | tCS  tFP  tAP tAPo | blks |"
           "  median    mean  stddev     min     max  | p2p       | iW/iR       tW/tR       | speedup\n");
    printf("  ---+-------+-------------------------+---------------------+------+"
           "------------------------------------------+-----------+-------------------------+---------\n");

    float bestMedian = 1e30f;
    int bestIdx = -1;
    unsigned refCount = 0;

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        if (!r.valid)
        {
            printf("  %2d | %5u | %3u %4u %4u %3u %4u | %3u %4u %4u %3u | %4s |"
                   " %-40s | --        | --                      | --\n",
                   r.numWarps, r.params.stackCap,
                   r.params.chunkSize, r.params.forcePush, r.params.attemptPush,
                   r.params.attemptPop, r.params.forcePop,
                   r.params.travChunkSize, r.params.travForcePush,
                   r.params.travAttemptPush, r.params.travAttemptPop,
                   "--", "SKIPPED");
            continue;
        }

        if (refCount == 0) refCount = r.p2pCount;
        const char* note = (r.p2pCount != refCount) ? " MISMATCH!" : "";

        float speedup = singleStats.median / r.stats.median;
        printf("  %2d | %5u | %3u %4u %4u %3u %4u | %3u %4u %4u %3u | %4d |"
               " %7.3f %7.3f %7.3f %7.3f %7.3f | %-9u | %6u/%-6u %6u/%-6u | %5.2fx%s\n",
               r.numWarps, r.params.stackCap,
               r.params.chunkSize, r.params.forcePush, r.params.attemptPush,
               r.params.attemptPop, r.params.forcePop,
               r.params.travChunkSize, r.params.travForcePush,
               r.params.travAttemptPush, r.params.travAttemptPop,
               r.totalBlocks,
               r.stats.median, r.stats.mean, r.stats.stddev,
               r.stats.minVal, r.stats.maxVal,
               r.p2pCount,
               r.iactWHead, r.iactRHead, r.travWHead, r.travRHead,
               speedup, note);

        if (r.stats.median < bestMedian)
        {
            bestMedian = r.stats.median;
            bestIdx = int(i);
        }
    }

    printf("  ---+-------+-------------------------+---------------------+------+"
           "------------------------------------------+-----------+-------------------------+---------\n");

    if (bestIdx >= 0)
    {
        const auto& b = results[bestIdx];
        float bestSpeedup = singleStats.median / b.stats.median;
        printf("\n  BEST:  nW=%d  SC=%u  iCS=%u iFP=%u iAP=%u iAPo=%u iFPo=%u  "
               "tCS=%u tFP=%u tAP=%u tAPo=%u  blks=%d  =>  median=%.3f ms  (%.2fx vs single)\n",
               b.numWarps, b.params.stackCap,
               b.params.chunkSize, b.params.forcePush, b.params.attemptPush,
               b.params.attemptPop, b.params.forcePop,
               b.params.travChunkSize, b.params.travForcePush,
               b.params.travAttemptPush, b.params.travAttemptPop,
               b.totalBlocks, b.stats.median, bestSpeedup);
    }
    printf("════════════════════════════════════════════════════════════════════════"
           "══════════════════════════════════════════════════════════════════════"
           "════════════════════════════════════\n");

    cudaFree(d_count);
}

TEST(Traversal, tuneP2PBenchmark)
{
    tuneP2PBenchmark(10000000, 16, 5, 20);
}

} // namespace cstone
