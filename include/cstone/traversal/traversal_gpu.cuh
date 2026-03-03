/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

 /*! @file
 * @brief Generic octree traversal methods
 *
 * @author Timo Schwab <tischwab@ethz.ch>
 */

#pragma once
#pragma nv_diag_suppress static_var_with_dynamic_init

#include "cstone/tree/octree.hpp"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/warpscan.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>

namespace cg = cooperative_groups;

__device__ __forceinline__ unsigned linear_block_rank()
{
    cg::grid_group grid = cg::this_grid();
    return grid.block_rank();
}

__device__ __forceinline__ unsigned blocks_per_cluster_runtime()
{
    cg::cluster_group cl = cg::this_cluster();
    return cl.num_blocks();
}

__device__ __forceinline__ unsigned block_rank_in_cluster()
{
    cg::cluster_group cl = cg::this_cluster();
    return cl.block_rank();
}

__device__ __forceinline__ unsigned cluster_rank_in_grid()
{
    return linear_block_rank() / blocks_per_cluster_runtime();
}

__device__ __forceinline__ unsigned num_clusters_runtime()
{
    return cg::this_grid().num_blocks() / blocks_per_cluster_runtime();
}

__device__ __forceinline__ unsigned thread_rank_in_block()
{
    return cg::this_thread_block().thread_rank();
}

__device__ __forceinline__ unsigned lane_id()
{
    unsigned r;
    asm("mov.u32 %0, %laneid;" : "=r"(r));
    return r;
}

HOST_DEVICE_FUN __forceinline__
constexpr std::size_t align_up(std::size_t x, std::size_t a)
{
    return (x + (a - 1)) & ~(a - 1);
}

template<class MAC>
__device__ __forceinline__
bool assignPairBySplitting_regress(const TreeNodeIndex* __restrict__ childOffsets,
                                  TreeNodeIndex& a, TreeNodeIndex& b,
                                  unsigned rid, unsigned N,
                                  unsigned& active_count,
                                  MAC&& continuation)
{
    if (N <= 1) { active_count = 1; return rid == 0; }

    const unsigned L_target = ceil_log8(N);
    unsigned digits[16];
    if (L_target > 16) { active_count = 1; return rid == 0; }
    decode_base8_digits(rid, L_target, digits);

    // Track last safe snapshot
    TreeNodeIndex a_safe = a, b_safe = b;
    unsigned L_safe = 0;               // number of split levels safely applied
    // unsigned digits_safe[16];          // prefix digits that were safely applied

    // Try to split up to L_target levels, but stop at first unsafe.
    for (unsigned level = 0; level < L_target; ++level)
    {
        if (!splitSafe(childOffsets, a, b, continuation))
        {
            // Regress to last safe state
            a = a_safe; b = b_safe;
            break;
        }

        const unsigned oct = digits[level];

        if (a < b) a = childOffsets[a] + (TreeNodeIndex)oct;
        else       b = childOffsets[b] + (TreeNodeIndex)oct;

        a_safe = a;
        b_safe = b;
        L_safe = level + 1;
    }

    const unsigned fanout = pow8(L_safe);
    active_count = (fanout < N) ? fanout : N;

    // If we couldn't safely split even once, fanout=1 -> only rid==0 active.
    return rid < active_count;
}

namespace cstone
{

__device__ __forceinline__ bool isLeaf(const TreeNodeIndex* __restrict__ childOffsets,
                                       TreeNodeIndex n)
{
    return childOffsets[n] == 0;
}

__device__ __forceinline__ unsigned warp_id_in_block()
{
    return threadIdx.x >> 5;
}


// ──────────────────────────────────────────────────────────────────────────────
// Block-level dual traversal (synchronized two-phase model)
//
// Two phases separated by block.sync():
//   Phase 1 (PRODUCE): All warps traverse. Leaf pairs append to flat buffers
//                       (single atomicAdd per warp, no committed/fence/backpressure).
//   Phase 2 (DRAIN):   All warps consume. Each warp assigned to either p2p or
//                       m2l (not both) — zero warp divergence.
//
// The shared traversal queue (for inter-warp load balancing) is kept as-is.
// ──────────────────────────────────────────────────────────────────────────────

template<int numWarps, unsigned queueCap, class MAC, class M2L, class P2P>
__device__ void dualTraversalBlock(
    const TreeNodeIndex* __restrict__ childOffsets,
    TreeNodeIndex a, TreeNodeIndex b,
    MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    extern __shared__ char dynSmem[];

    cg::thread_block block = cg::this_thread_block();
    unsigned tid     = block.thread_rank();
    unsigned warpId  = tid / GpuConfig::warpSize;
    unsigned laneIdx = tid % GpuConfig::warpSize;

    // Handle trivial case: both are leaves
    if (isLeaf(childOffsets, a) && isLeaf(childOffsets, b)) {
        if (tid == 0) {
            if (continuation(a, b)) { p2p(a, b); }
        }
        return;
    } else if (isLeaf(childOffsets, a) || isLeaf(childOffsets, b)) {
        if (tid == 0) {
            if(!continuation(a,b)) { m2l(a,b); }
        }
        return;
    }


    
}

// ──────────────────────────────────────────────────────────────────────────────
// TBC level: split work across blocks within a cluster
// ──────────────────────────────────────────────────────────────────────────────

template <int numWarps, unsigned queueCap, class MAC, class M2L, class P2P>
__device__ void dualTraversalTBC(const TreeNodeIndex* __restrict__ childOffsets,
                                 TreeNodeIndex a, TreeNodeIndex b,
                                 MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    const unsigned block_in_cluster = block_rank_in_cluster();
    const unsigned blocksPerCluster = blocks_per_cluster_runtime();

    unsigned active_blocks = 0;
    const bool active_block =
        assignPairByBalancedSplit(childOffsets, a, b,
                                      block_in_cluster, blocksPerCluster,
                                      active_blocks,
                                      std::forward<MAC>(continuation));

    if (!active_block) return;

    dualTraversalBlock<numWarps, queueCap>(childOffsets, a, b,
                                            std::forward<MAC>(continuation),
                                            std::forward<M2L>(m2l),
                                            std::forward<P2P>(p2p));
}

// ──────────────────────────────────────────────────────────────────────────────
// GPU level: split work across clusters
// ──────────────────────────────────────────────────────────────────────────────

template<int numWarps, unsigned queueCap, class MAC, class M2L, class P2P>
__device__ void dualTraversalGPU(const TreeNodeIndex* __restrict__ childOffsets,
                                 TreeNodeIndex rootA, TreeNodeIndex rootB,
                                 MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    const unsigned cluster_id  = cluster_rank_in_grid();
    const unsigned numClusters = num_clusters_runtime();

    TreeNodeIndex a = rootA, b = rootB;

    unsigned active_clusters = 0;
    const bool active_cluster =
        assignPairByBalancedSplit(childOffsets, a, b,
                                      cluster_id, numClusters,
                                      active_clusters,
                                      std::forward<MAC>(continuation));

    if (!active_cluster) return;

    dualTraversalTBC<numWarps, queueCap>(childOffsets, a, b,
                                          std::forward<MAC>(continuation),
                                          std::forward<M2L>(m2l),
                                          std::forward<P2P>(p2p));
}

} // namespace cstone
