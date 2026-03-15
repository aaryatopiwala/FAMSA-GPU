#include "GPU_LCS.cuh"
#include <vector>
#include <stdint.h>
#include <algorithm>
#include <stdio.h>
#include <cstring>
#include <fstream>
#include <string>
#include <iostream>
#include "../core/sequence.h"
#include <cuda_runtime.h>
#include <limits>
#include <math.h>


// __global__ void LCS_Kernel(
//     const symbol_t* d_concat_seqs,
//     //const symbol_t* d_ref,
//     const uint64_t* d_ref_bitmasks,
//     uint64_t* d_carry,
//     uint64_t* d_workspace,
//     int bv_len,
//     int chunk_size,
//     int iteration_step_size,
//     const int* d_offsets,
//     const int* d_lengths,
//     uint32_t* d_out_lcs,
//     int batch_n)
// {
//     int gs = gridDim.x;
//     int bs = blockDim.x;
//     int bx = blockIdx.x;
//     int tx = threadIdx.x;

//     extern __shared__ uint64_t shared_mem[];
//     uint64_t* s_ref_bitmasks = shared_mem;                          // shared memory for reference bitmasks, size will be NO_SYMBOLS * bv_len
//     uint64_t* s_workspace    = shared_mem + NO_SYMBOLS * bv_len;    // shared memory workspace for bit vectors of the current row
//     uint64_t* s_carry        = s_workspace + bv_len;                // shared memory for carry bits for the current wavefront step

//     // Load reference bitmasks into shared memory for faster access for each block
//     if (tx < NO_SYMBOLS * bv_len) {
//         for (int i = tx; i < NO_SYMBOLS * bv_len; i+=bs) {
//             s_ref_bitmasks[i] = d_ref_bitmasks[i];
//         }   
//     }
//     __syncthreads(); // Wait for reference bitmasks to be loaded into shared memory

//     // serial method that will be parallelized later
//     // if (tx == 0 && bx == 0){
//     //     for (int i = 0; i < batch_n; ++i) {
//     //         const symbol_t* seq = d_concat_seqs + d_offsets[i];
//     //         int seq_len = d_lengths[i];
//     //         int ref_len = 0; // need to pass this in or compute from d_ref
            
//     //         for (int w = 0; w < bv_len; ++w) {
//     //             s_workspace[w] = ~((uint64_t)0); // initialize workspace
//     //         }

//     //         for (int j = 0; j < seq_len; ++j) {
//     //             unsigned char c = seq[j];
//     //             if ( c == 22 || c >= 32) { // unknown symbol, or over symbol range, skip
//     //                 continue;
//     //             }
//     //             const uint64_t* s0b = s_ref_bitmasks + c * bv_len; // bitmask for symbol c
//     //             uint64_t carry = 0;
//     //             for (int w = 0; w < bv_len; ++w) {
//     //                 uint64_t V = s_workspace[w];
//     //                 uint64_t tb = V & s0b[w];
//     //                 uint64_t V2 = V + tb + carry;
//     //                 carry = (V2 < V) ? 1 : 0; // detect overflow
//     //                 s_workspace[w] = V2 | (V - tb);
//     //             }
//     //         }
            
//     //         uint32_t res = 0;
//     //         for (int w = 0; w < bv_len; ++w) {
//     //             res += __popcll(~s_workspace[w]); // count set bits in ~workspace
//     //         }
//     //         d_out_lcs[i] = res; 
//     //     }
//     // }

//     // if (bx == 0){
//         // for (int i = 0; i < batch_n; ++i) {
//         for (int b = bx; b < batch_n; b += gs) {
//             const symbol_t* seq = d_concat_seqs + d_offsets[b];
//             int seq_len = d_lengths[b];
//             int ref_len = 0; // need to pass this in or compute from d_ref
            
//             if (tx < bv_len) {
//                 for (int w = tx; w < bv_len; w+=bs) {
//                     s_workspace[w] = ~((uint64_t)0); // initialize workspace, set bit vector of each chunk to all 1s
//                 }
//             }
//             __syncthreads();

//             int num_steps = (seq_len + iteration_step_size - 1) / iteration_step_size;
//             if (tx < num_steps){
//                 for (int n = tx; n < bv_len * num_steps; n += bs){
//                     s_carry[n] = 0;
//                 }
//             }
//             __syncthreads(); // Wait for the whole row to be loaded

//            for (int k = 0; k < (bv_len + num_steps - 1); ++k) {
//                 int i_min = max(0, k - (num_steps - 1));
//                 int i_max = min(bv_len - 1, k);

//                 for (int row = i_min + tx; row <= i_max; row += bs) {
//                     int i = row;
//                     int j = k - row;

//                     uint64_t carry = s_carry[i * num_steps + j];
//                     unsigned char c = seq[j];
//                     if (c != 22 && c < 32) {
//                         const uint64_t* s0b = s_ref_bitmasks + c * bv_len;
//                         uint64_t V = s_workspace[i];
//                         uint64_t tb = V & s0b[i];
//                         uint64_t sum1 = V + tb;
//                         uint64_t V2 = sum1 + carry;
//                         s_workspace[i] = V2 | (V - tb);
//                         carry = (sum1 < V) || (V2 < sum1);
//                     }
//                     if (i + 1 < bv_len)
//                         s_carry[(i + 1) * num_steps + j] = carry;
//                 }
//                 __syncthreads();
//             }

//             if (tx == 0) {
//                 uint32_t res = 0;
//                 for (int w = 0; w < bv_len; ++w) {
//                     res += __popcll(~s_workspace[w]);
//                 }
//                 d_out_lcs[b] = res;
//             }
//         __syncthreads();
//         }

//     // }
// }

__global__ void LCS_Kernel_ThreadPerSeq(
    const symbol_t* __restrict__ d_concat_seqs,
    const uint64_t* __restrict__ d_ref_bitmasks,
    const int* __restrict__ d_offsets,
    const int* __restrict__ d_lengths,
    uint32_t* __restrict__ d_out_lcs,
    int bv_len,
    int batch_n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int b = tid; b < batch_n; b += stride) {
        const symbol_t* seq = d_concat_seqs + d_offsets[b];
        int seq_len = d_lengths[b];

        // bit vector
        uint64_t V_local[4];


        for (int w = 0; w < bv_len; ++w) {
            V_local[w] = ~0ull;
        }

        for (int j = 0; j < seq_len; ++j) {
            unsigned char c = seq[j];

            if (c == 22 || c >= 32) {
                continue;
            }

            const uint64_t* s0b = d_ref_bitmasks + ((int)c) * bv_len;
            uint64_t carry = 0;

            for (int w = 0; w < bv_len; ++w) {
                uint64_t V  = V_local[w];
                uint64_t tb = V & s0b[w];

                uint64_t sum1 = V + tb;
                uint64_t V2 = sum1 + carry;

                carry = (sum1 < V) || (V2 < sum1);
                V_local[w] = V2 | (V - tb);
            }
        }

        uint32_t res = 0;
        for (int w = 0; w < bv_len; ++w) {
            res += __popcll(~V_local[w]);
        }

        d_out_lcs[b] = res;
    }
}


__global__ void LCS_Kernel_WarpPerSeq(
    const symbol_t* __restrict__ d_concat_seqs,
    const uint64_t* __restrict__ d_ref_bitmasks,
    const int* __restrict__ d_offsets,
    const int* __restrict__ d_lengths,
    uint32_t* __restrict__ d_out_lcs,
    int bv_len,
    int batch_n)
{
    extern __shared__ uint64_t s_bitmasks[];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_in_block = tid >> 5;
    const int warps_per_block = blockDim.x >> 5;

    const int global_warp_id = blockIdx.x * warps_per_block + warp_in_block;
    const int total_warps = gridDim.x * warps_per_block;

    const unsigned FULL_MASK = 0xffffffffu;
    const unsigned ACTIVE_MASK = (bv_len == 32) ? 0xffffffffu : ((1u << bv_len) - 1u);

    for (int i = tid; i < NO_SYMBOLS * bv_len; i += blockDim.x) {
        s_bitmasks[i] = d_ref_bitmasks[i];
    }
    __syncthreads();

    for (int b = global_warp_id; b < batch_n; b += total_warps) {
        const symbol_t* seq = d_concat_seqs + d_offsets[b];
        const int seq_len = d_lengths[b];
        uint64_t V = (lane < bv_len) ? ~0ull : 0ull;

        for (int j = 0; j < seq_len; ++j) {
            // symbol_t c = 0;
            // if (lane == 0) c = seq[j];
            // c = __shfl_sync(FULL_MASK, c, 0);
            symbol_t c = seq[j];
            if (c == 22 || c >= 32) {
                continue;
            }

            uint64_t mask_word = (lane < bv_len)
                ? s_bitmasks[(int)c * bv_len + lane]
                : 0ull;

            uint64_t tb  = V & mask_word;
            uint64_t sum1  = V + tb;
            uint64_t minus = V - tb;

            // carry_out = g | (p & carry_in)
            unsigned g = 0u;
            unsigned p = 0u;
            if (lane < bv_len) {
                g = (sum1 < V) ? 1u : 0u;
                p = (sum1 == 0xffffffffffffffffull) ? 1u : 0u;
            }

            unsigned gmask = __ballot_sync(FULL_MASK, g);
            unsigned pmask = __ballot_sync(FULL_MASK, p);

            gmask &= ACTIVE_MASK;
            pmask &= ACTIVE_MASK;

            unsigned G = gmask;
            unsigned P = pmask;

            G |= P & (G << 1);
            P &= (P << 1);

            G |= P & (G << 2);
            P &= (P << 2);

            G |= P & (G << 4);
            P &= (P << 4);

            G |= P & (G << 8);
            P &= (P << 8);

            G |= P & (G << 16);

            // lane_i carry_in is carry-out of lane i-1
            unsigned carry_mask = (G << 1) & ACTIVE_MASK;
            unsigned carry_in   = (lane < bv_len) ? ((carry_mask >> lane) & 1u) : 0u;

            if (lane < bv_len) {
                uint64_t V2 = sum1 + (uint64_t)carry_in;
                V = V2 | minus;
            }
        }
        uint32_t local_pop = (lane < bv_len) ? __popcll(~V) : 0u;


        local_pop += __shfl_down_sync(FULL_MASK, local_pop, 16);
        local_pop += __shfl_down_sync(FULL_MASK, local_pop, 8);
        local_pop += __shfl_down_sync(FULL_MASK, local_pop, 4);
        local_pop += __shfl_down_sync(FULL_MASK, local_pop, 2);
        local_pop += __shfl_down_sync(FULL_MASK, local_pop, 1);

        if (lane == 0) {
            d_out_lcs[b] = local_pop;
        }
    }
}

__global__ void LCS_Kernel_BlockSerial(
    const symbol_t* d_concat_seqs,
    const uint64_t* d_ref_bitmasks,
    const int* d_offsets,
    const int* d_lengths,
    uint32_t* d_out_lcs,
    int bv_len,
    int batch_n)
{
    int gs = gridDim.x;
    int bs = blockDim.x;
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    extern __shared__ uint64_t shared_mem[];
    uint64_t* s_ref_bitmasks = shared_mem;
    uint64_t* s_workspace = shared_mem + NO_SYMBOLS * bv_len;

    for (int i = tx; i < NO_SYMBOLS * bv_len; i += bs) {
        s_ref_bitmasks[i] = d_ref_bitmasks[i];
    }
    __syncthreads();

    for (int b = bx; b < batch_n; b += gs) {
        const symbol_t* seq = d_concat_seqs + d_offsets[b];
        int seq_len = d_lengths[b];

        for (int w = tx; w < bv_len; w += bs) {
            s_workspace[w] = ~((uint64_t)0);
        }
        __syncthreads();

        // process sequence positions
        if (tx == 0) {
            for (int j = 0; j < seq_len; ++j) {
                unsigned char c = seq[j];

                // invalid
                if (c == 22 || c >= 32) {
                    continue;
                }

                const uint64_t* s0b = s_ref_bitmasks + c * bv_len;
                uint64_t carry = 0;

                for (int w = 0; w < bv_len; ++w) {
                    uint64_t V  = s_workspace[w];
                    uint64_t tb = V & s0b[w];

                    uint64_t sum1 = V + tb;
                    uint64_t V2   = sum1 + carry;

                    carry = (sum1 < V) || (V2 < sum1);
                    s_workspace[w] = V2 | (V - tb);
                }
            }

            uint32_t res = 0;
            for (int w = 0; w < bv_len; ++w) {
                res += __popcll(~s_workspace[w]);
            }
            d_out_lcs[b] = res;
        }
        __syncthreads();
    }
}

struct GpuLCSContext {
    // Change N_STREAMS and Batch size here
    static constexpr int BATCH_SIZE = 4000;
    static constexpr int N_STREAMS  = 2;

    int*         d_offset_pool        = nullptr;
    int*         h_offset_pool_pinned = nullptr;
    int*         d_length_pool        = nullptr;
    int*         h_length_pool_pinned = nullptr;
    uint32_t*    d_out_pool           = nullptr;
    uint32_t*    h_out_pool_pinned    = nullptr;
    symbol_t*    d_concat_pool        = nullptr;
    symbol_t*    h_concat_pool_pinned = nullptr;
    size_t       alloc_concat_len     = 0;
    cudaStream_t streams[N_STREAMS]   = {};
    cudaEvent_t  events[N_STREAMS]    = {};
    bool         initialized          = false;

    uint64_t*    d_ref_bitmasks   = nullptr;
    int          cached_bv_len    = -1;
    const void*  cached_ref_ptr   = nullptr;

    bool init() {
        cudaError_t err;
        err = cudaMalloc    (&d_offset_pool,        BATCH_SIZE * sizeof(int)      * N_STREAMS); if (err != cudaSuccess) return fail(err);
        err = cudaMallocHost(&h_offset_pool_pinned, BATCH_SIZE * sizeof(int)      * N_STREAMS); if (err != cudaSuccess) return fail(err);
        err = cudaMalloc    (&d_length_pool,        BATCH_SIZE * sizeof(int)      * N_STREAMS); if (err != cudaSuccess) return fail(err);
        err = cudaMallocHost(&h_length_pool_pinned, BATCH_SIZE * sizeof(int)      * N_STREAMS); if (err != cudaSuccess) return fail(err);
        err = cudaMalloc    (&d_out_pool,           BATCH_SIZE * sizeof(uint32_t) * N_STREAMS); if (err != cudaSuccess) return fail(err);
        err = cudaMallocHost(&h_out_pool_pinned,    BATCH_SIZE * sizeof(uint32_t) * N_STREAMS); if (err != cudaSuccess) return fail(err);
        for (int i = 0; i < N_STREAMS; ++i) {
            err = cudaStreamCreate(&streams[i]);                                 if (err != cudaSuccess) return fail(err);
            err = cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);  if (err != cudaSuccess) return fail(err);
        }
        initialized = true;
        return true;
    }

    bool ensureConcatPool(size_t needed) {
        if (needed <= alloc_concat_len) return true;
        if (d_concat_pool)        { cudaFree(d_concat_pool);            d_concat_pool        = nullptr; }
        if (h_concat_pool_pinned) { cudaFreeHost(h_concat_pool_pinned); h_concat_pool_pinned = nullptr; }
        cudaError_t err;
        err = cudaMalloc    (&d_concat_pool,        needed * sizeof(symbol_t) * N_STREAMS); if (err != cudaSuccess) return fail(err);
        err = cudaMallocHost(&h_concat_pool_pinned, needed * sizeof(symbol_t) * N_STREAMS); if (err != cudaSuccess) return fail(err);
        alloc_concat_len = needed;
        return true;
    }

    bool ensureRefBitmasks(const void* ref_ptr, const void* src, int bv_len_in) {
        if (ref_ptr == cached_ref_ptr && bv_len_in == cached_bv_len)
            return true;

        if (d_ref_bitmasks) { cudaFree(d_ref_bitmasks); d_ref_bitmasks = nullptr; }

        cudaError_t err;
        err = cudaMalloc(&d_ref_bitmasks, (size_t)NO_SYMBOLS * bv_len_in * sizeof(uint64_t));
        if (err != cudaSuccess) return fail(err);
        err = cudaMemcpy(d_ref_bitmasks, src,
                        (size_t)NO_SYMBOLS * bv_len_in * sizeof(uint64_t),
                        cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { cudaFree(d_ref_bitmasks); d_ref_bitmasks = nullptr; return fail(err); }

        cached_ref_ptr = ref_ptr;
        cached_bv_len  = bv_len_in;
        return true;
    }

    ~GpuLCSContext() {
        if (!initialized) return;
        for (int i = 0; i < N_STREAMS; ++i) {
            cudaStreamDestroy(streams[i]);
            cudaEventDestroy(events[i]);
        }
        if (d_ref_bitmasks)       cudaFree(d_ref_bitmasks);
        if (d_concat_pool)        cudaFree(d_concat_pool);
        if (h_concat_pool_pinned) cudaFreeHost(h_concat_pool_pinned);
        if (d_offset_pool)        cudaFree(d_offset_pool);
        if (h_offset_pool_pinned) cudaFreeHost(h_offset_pool_pinned);
        if (d_length_pool)        cudaFree(d_length_pool);
        if (h_length_pool_pinned) cudaFreeHost(h_length_pool_pinned);
        if (d_out_pool)           cudaFree(d_out_pool);
        if (h_out_pool_pinned)    cudaFreeHost(h_out_pool_pinned);
    }

private:
    static bool fail(cudaError_t err) {
        fprintf(stderr, "GPU_ERROR (GpuLCSContext): %s (%s)\n",
                cudaGetErrorString(err), cudaGetErrorName(err));
        return false;
    }
};

static thread_local GpuLCSContext tl_ctx;

void GpuLCS::computeLCSLengths(
        CSequence*  ref,
        CSequence** sequences,
        int         n_seqs,
        uint32_t*   out_vector,
        CLCSBP&     lcsbp)
{
    GpuLCSContext& ctx = tl_ctx;

    auto pref  = ref;
    int bv_len = (pref->data_size + 63) / 64;

    if (!ctx.initialized) {
        if (!ctx.init()) return;
    }

    if (!ctx.ensureRefBitmasks(pref->p_bit_masks, pref->p_bit_masks, bv_len)) return;

    size_t max_batch_symbols = 0;
    for (int start = 0; start < n_seqs; start += ctx.BATCH_SIZE) {
        size_t cur = 0;
        int end = std::min(start + ctx.BATCH_SIZE, n_seqs);
        for (int i = start; i < end; ++i)
            cur += sequences[i]->length;
        max_batch_symbols = std::max(max_batch_symbols, cur);
    }
    if (!ctx.ensureConcatPool(max_batch_symbols)) return;

    const size_t stream_concat_len = max_batch_symbols;
    const size_t stream_offset_len = ctx.BATCH_SIZE;

    bool stream_busy[ctx.N_STREAMS]        = {};
    int  stream_batch_start[ctx.N_STREAMS] = {};
    int  stream_batch_n[ctx.N_STREAMS]     = {};

    for (int start = 0; start < n_seqs; start += ctx.BATCH_SIZE) {

        int batch_n   = std::min(ctx.BATCH_SIZE, n_seqs - start);
        int si        = (start / ctx.BATCH_SIZE) % ctx.N_STREAMS;
        cudaStream_t currStream = ctx.streams[si];

        symbol_t* d_concat_seqs    = ctx.d_concat_pool        + si * stream_concat_len;
        symbol_t* h_concat_pinned  = ctx.h_concat_pool_pinned + si * stream_concat_len;
        uint32_t* d_out_lcs        = ctx.d_out_pool           + si * stream_offset_len;
        uint32_t* h_out_lcs_pinned = ctx.h_out_pool_pinned    + si * stream_offset_len;
        int*      d_offsets        = ctx.d_offset_pool        + si * stream_offset_len;
        int*      h_offsets_pinned = ctx.h_offset_pool_pinned + si * stream_offset_len;
        int*      d_lengths        = ctx.d_length_pool        + si * stream_offset_len;
        int*      h_lengths_pinned = ctx.h_length_pool_pinned + si * stream_offset_len;

        if (stream_busy[si]) {
            cudaEventSynchronize(ctx.events[si]);
            int ps = stream_batch_start[si];
            int pn = stream_batch_n[si];
            for (int i = 0; i < pn; ++i)
                out_vector[ps + i] = h_out_lcs_pinned[i];
            stream_busy[si] = false;
        }

        symbol_t* write_ptr = h_concat_pinned;
        for (int i = 0; i < batch_n; ++i) {
            auto pseq = sequences[start + i];
            h_lengths_pinned[i] = pseq->length;
            h_offsets_pinned[i] = (int)(write_ptr - h_concat_pinned);
            memcpy(write_ptr, pseq->data, pseq->length * sizeof(symbol_t));
            write_ptr += pseq->length;
        }
        size_t total_symbols = (size_t)(write_ptr - h_concat_pinned);

        cudaError_t err;
        err = cudaMemcpyAsync(d_concat_seqs,  h_concat_pinned,  total_symbols * sizeof(symbol_t), cudaMemcpyHostToDevice, currStream);
        if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err)); return; }
        err = cudaMemcpyAsync(d_offsets,      h_offsets_pinned, batch_n * sizeof(int),            cudaMemcpyHostToDevice, currStream);
        if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err)); return; }
        err = cudaMemcpyAsync(d_lengths,      h_lengths_pinned, batch_n * sizeof(int),            cudaMemcpyHostToDevice, currStream);
        if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err)); return; }

        if (bv_len <= 4) {
            // Thread kernel here
            LCS_Kernel_ThreadPerSeq<<<64, 128, 0, currStream>>>(
                d_concat_seqs, ctx.d_ref_bitmasks, d_offsets, d_lengths,
                d_out_lcs, bv_len, batch_n);
        } else if (bv_len <= 32) {
            // Warp kernel here
            size_t smem = NO_SYMBOLS * bv_len * sizeof(uint64_t);
            LCS_Kernel_WarpPerSeq<<<128, 128, smem, currStream>>>(
                d_concat_seqs, ctx.d_ref_bitmasks, d_offsets, d_lengths,
                d_out_lcs, bv_len, batch_n);
        } else {
            // Block kernel here 
            size_t smem = (NO_SYMBOLS * bv_len + bv_len) * sizeof(uint64_t);
            LCS_Kernel_BlockSerial<<<1024, 32, smem, currStream>>>(
                d_concat_seqs, ctx.d_ref_bitmasks, d_offsets, d_lengths,
                d_out_lcs, bv_len, batch_n);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR after kernel launch: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
            return;
        }

        err = cudaMemcpyAsync(h_out_lcs_pinned, d_out_lcs, batch_n * sizeof(uint32_t), cudaMemcpyDeviceToHost, currStream);
        if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err)); return; }

        cudaEventRecord(ctx.events[si], currStream);
        stream_busy[si]        = true;
        stream_batch_start[si] = start;
        stream_batch_n[si]     = batch_n;
    }

    for (int i = 0; i < ctx.N_STREAMS; ++i) {
        if (stream_busy[i]) {
            cudaEventSynchronize(ctx.events[i]);
            int ps = stream_batch_start[i];
            int pn = stream_batch_n[i];
            uint32_t* h_out = ctx.h_out_pool_pinned + i * stream_offset_len;
            for (int j = 0; j < pn; ++j)
                out_vector[ps + j] = h_out[j];
        }
    }
}

// void GpuLCS::computeLCSLengths(
// 		CSequence* ref,
// 		CSequence** sequences,
// 		int n_seqs,
// 		uint32_t* out_vector,
// 		CLCSBP& lcsbp)
// {

//     // 1. bitmasks are already in ref
//     const int BATCH_SIZE = 4000;
//     auto pref = ref;
//     int bv_len = (pref->data_size + 63) / 64;


//     // find the longest possible length among batches to use for concat, also max sequence length
//     int max_batch_len = 0;
//     int max_seq_len = 0;
//     for (int start = 0; start < n_seqs; start += BATCH_SIZE) {
//         size_t cur = 0;
//         int end = min(start + BATCH_SIZE, n_seqs);
//         for (int i = start; i < end; ++i) {
//             cur += sequences[i]->length;
//             max_seq_len = max(max_seq_len, sequences[i]->length);
//         }
//         max_batch_len = max(max_batch_len, (int)cur);
//     }
//     size_t stream_concat_len = max_batch_len;
//     size_t stream_offset_len = BATCH_SIZE;
//     size_t stream_output_len = BATCH_SIZE * sizeof(uint32_t);

//     std::vector<uint64_t> h_ref_bitmasks((size_t)(NO_SYMBOLS) * bv_len);

//     for (int j = 0; j < NO_SYMBOLS * bv_len; ++j) {
//         h_ref_bitmasks[j] = pref->p_bit_masks[j];
//     }

//     uint64_t* d_ref_bitmasks;
//     cudaError_t err = cudaMalloc(&d_ref_bitmasks, h_ref_bitmasks.size() * sizeof(uint64_t));    
//     err = cudaMemcpy(d_ref_bitmasks, h_ref_bitmasks.data(), h_ref_bitmasks.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
//     if (err != cudaSuccess) {
//         fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
//         return;
//     }

//     int nStreams = 1;
//     std::vector<cudaStream_t> streams(nStreams);
//     std::vector<cudaEvent_t> events(nStreams);
//     for (int i = 0; i < nStreams; ++i) {
//         err = cudaStreamCreate(&streams[i]);
//         if (err != cudaSuccess) {
//             fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
//             return;
//         }
//         err = cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
//         if (err != cudaSuccess) {
//             fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
//             return;
//         }
//     }

//     // pool since we allocate once and split it amongst async streams
//     symbol_t* d_concat_pool;
//     symbol_t* h_concat_pool_pinned;
//     int* d_offset_pool;
//     int* h_offset_pool_pinned;
//     int* d_length_pool;
//     int* h_length_pool_pinned;
//     uint32_t* d_out_pool;
//     uint32_t* h_out_pool_pinned;

//     err = cudaMalloc(&d_concat_pool, stream_concat_len * sizeof(symbol_t) * nStreams);
//     err = cudaMallocHost(&h_concat_pool_pinned, stream_concat_len * sizeof(symbol_t) * nStreams);

//     err = cudaMalloc(&d_offset_pool, stream_offset_len * sizeof(int) * nStreams);
//     err = cudaMallocHost(&h_offset_pool_pinned, stream_offset_len * sizeof(int) * nStreams);

//     err = cudaMalloc(&d_length_pool, stream_offset_len * sizeof(int) * nStreams);
//     err = cudaMallocHost(&h_length_pool_pinned, stream_offset_len * sizeof(int) * nStreams);

//     err = cudaMalloc(&d_out_pool, stream_output_len * nStreams);
//     err = cudaMallocHost(&h_out_pool_pinned, stream_output_len * nStreams);

//     if (err != cudaSuccess) {
//         fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
//         return;
//     }

//     std::vector<bool> stream_busy(nStreams, false); // track which streams are busy
//     std::vector<int> stream_batch_start(nStreams, 0); // track which batch is in which stream
//     std::vector<int> stream_batch_n(nStreams, 0); // track batch size for each stream

//     // 2. Divide sequences into batches and send to GPU
//     for (int start = 0; start < n_seqs; start += BATCH_SIZE) {
     
//         int batch_n = std::min(BATCH_SIZE, n_seqs - start); // batch is smmaller at the end
//         int batch_idx = start / BATCH_SIZE;
//         int stream_idx = batch_idx % nStreams;
//         cudaStream_t currStream = streams[stream_idx];

//         std::vector<symbol_t> h_concat_seqs;
//         std::vector<int> h_offsets(batch_n);
//         std::vector<int> h_lengths(batch_n);
//         int running_offset = 0;
//         for (int i = 0; i < batch_n; ++i) {
//             auto pseq = sequences[start + i];
//             int len = pseq->length;
//             h_lengths[i] = len;
//             h_offsets[i] = running_offset;
//             h_concat_seqs.insert(h_concat_seqs.end(), pseq->data, pseq->data + len);
//             running_offset += len;
//         }

//         // per stream slice of pool
//         symbol_t* d_concat_seqs = d_concat_pool + stream_idx * stream_concat_len;
//         symbol_t* h_concat_seqs_pinned = h_concat_pool_pinned + stream_idx * stream_concat_len;
//         int* d_offsets = d_offset_pool + stream_idx * stream_offset_len;
//         int* h_offsets_pinned = h_offset_pool_pinned + stream_idx * stream_offset_len;
//         int* d_lengths = d_length_pool + stream_idx * stream_offset_len;
//         int* h_lengths_pinned = h_length_pool_pinned + stream_idx * stream_offset_len;
//         uint32_t* d_out_lcs = d_out_pool + stream_idx * stream_output_len/sizeof(uint32_t);
//         uint32_t* h_out_lcs_pinned = h_out_pool_pinned + stream_idx * stream_output_len/sizeof(uint32_t);
        
//         // check if stream is busy

//         if (stream_busy[stream_idx]) {
//             // wait for stream to finish
//             cudaEventSynchronize(events[stream_idx]);
//             // copy results back to output vector since async D2H is done
//             int prev_start = stream_batch_start[stream_idx];
//             int prev_n = stream_batch_n[stream_idx];
//             for (int i = 0; i < prev_n; ++i) {
//                 out_vector[prev_start + i] = h_out_pool_pinned[stream_idx * stream_output_len / sizeof(uint32_t) + i];
//             }
//             stream_busy[stream_idx] = false;
//         }

//         size_t concat_size = h_concat_seqs.size() * sizeof(symbol_t);
//         memcpy(h_concat_seqs_pinned, h_concat_seqs.data(), concat_size);
//         memcpy(h_offsets_pinned, h_offsets.data(), batch_n * sizeof(int));
//         memcpy(h_lengths_pinned, h_lengths.data(), batch_n * sizeof(int));

//         // copy data to GPU asynchronously
//         err = cudaMemcpyAsync(d_concat_seqs, h_concat_seqs_pinned, h_concat_seqs.size() * sizeof(symbol_t), cudaMemcpyHostToDevice, currStream);
//         //err = cudaMemcpyAsync(d_ref, h_ref_pinned, ref_len * sizeof(symbol_t), cudaMemcpyHostToDevice, currStream);
//         err = cudaMemcpyAsync(d_offsets, h_offsets_pinned, batch_n * sizeof(int), cudaMemcpyHostToDevice, currStream);
//         err = cudaMemcpyAsync(d_lengths, h_lengths_pinned, batch_n * sizeof(int), cudaMemcpyHostToDevice, currStream);
//         if (err != cudaSuccess) {
//             fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err)) ;
//             return;
//         }

        
//         // 3. do kernel
//         // int numBlocks = 1024;
//         // int blockSize = 64;
        
//         // size_t smem_bytes = (NO_SYMBOLS * bv_len + bv_len) * sizeof(uint64_t);
//         // LCS_Kernel_BlockSerial<<<numBlocks, blockSize, smem_bytes, currStream>>>(
//         //     d_concat_seqs,
//         //     d_ref_bitmasks,
//         //     d_offsets,
//         //     d_lengths,
//         //     d_out_lcs,
//         //     bv_len,
//         //     batch_n);

//         // if (bv_len <= 32) {
//         //     int blockSize = 128;
//         //     int numBlocks = 1024;
            
//         //     LCS_Kernel_WarpPerSeq<<<numBlocks, blockSize, 0, currStream>>>(
//         //         d_concat_seqs,
//         //         d_ref_bitmasks,
//         //         d_offsets,
//         //         d_lengths,
//         //         d_out_lcs,
//         //         bv_len,
//         //         batch_n);
//         // } else {
//         //     int blockSize = 128;
//         //     int numBlocks = 1024;

//         //     size_t smem_bytes = (NO_SYMBOLS * bv_len + bv_len) * sizeof(uint64_t);
//         //     LCS_Kernel_BlockSerial<<<numBlocks, blockSize, smem_bytes, currStream>>>(
//         //         d_concat_seqs,
//         //         d_ref_bitmasks,
//         //         d_offsets,
//         //         d_lengths,
//         //         d_out_lcs,
//         //         bv_len,
//         //         batch_n);
//         // }

//         if (bv_len <= 4) {
//             // Small-width
//             int blockSize = 256;
//             int numBlocks = 64;
//             printf("A");

//             LCS_Kernel_ThreadPerSeq<<<numBlocks, blockSize, 0, currStream>>>(
//                 d_concat_seqs,
//                 d_ref_bitmasks,
//                 d_offsets,
//                 d_lengths,
//                 d_out_lcs,
//                 bv_len,
//                 batch_n);
//         }
//         else if (bv_len <= 32) {
//             // Medium-width case
//             int blockSize = 128;
//             int numBlocks = 1024;
//             printf("B");

//             size_t smem_bytes = (NO_SYMBOLS * bv_len) * sizeof(uint64_t);

//             LCS_Kernel_WarpPerSeq<<<numBlocks, blockSize, smem_bytes, currStream>>>(
//                 d_concat_seqs,
//                 d_ref_bitmasks,
//                 d_offsets,
//                 d_lengths,
//                 d_out_lcs,
//                 bv_len,
//                 batch_n);
//         }
//         else {
//             // Large-width
//             int blockSize = 32;
//             int numBlocks = 1024;
//             printf("C");

//             size_t smem_bytes = (NO_SYMBOLS * bv_len + bv_len) * sizeof(uint64_t);

//             LCS_Kernel_BlockSerial<<<numBlocks, blockSize, smem_bytes, currStream>>>(
//                 d_concat_seqs,
//                 d_ref_bitmasks,
//                 d_offsets,
//                 d_lengths,
//                 d_out_lcs,
//                 bv_len,
//                 batch_n);
//         }


//         err = cudaGetLastError();
//         if (err != cudaSuccess) {
//             fprintf(stderr, "GPU_ERROR after kernel launch: %s (%s)\n",
//                     cudaGetErrorString(err), cudaGetErrorName(err));
//             return;
//         }
//         // 4. gpu to host

//         err = cudaMemcpyAsync(h_out_lcs_pinned, d_out_lcs, batch_n * sizeof(uint32_t), cudaMemcpyDeviceToHost, currStream);
//         if (err != cudaSuccess) {
//             fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
//             return;
//         }

//         cudaEventRecord(events[stream_idx], currStream);
//         stream_busy[stream_idx] = true;
//         stream_batch_start[stream_idx] = start;
//         stream_batch_n[stream_idx] = batch_n;

    
//     }

//     // drain remaining streams
//     for (int i = 0; i < nStreams; ++i) {
//         if (stream_busy[i]) {
//             cudaEventSynchronize(events[i]);
//             int prev_start = stream_batch_start[i];
//             int prev_n = stream_batch_n[i];
//             uint32_t* host_slice = h_out_pool_pinned + i * stream_output_len / sizeof(uint32_t);
//             for (int j = 0; j < prev_n; ++j) {
//                 out_vector[prev_start + j] = host_slice[j];
//             }
//         }
//     }

//     cudaFree(d_ref_bitmasks);
//     cudaFree(d_concat_pool);
//     cudaFreeHost(h_concat_pool_pinned);
//     cudaFree(d_offset_pool);
//     cudaFreeHost(h_offset_pool_pinned);
//     cudaFree(d_length_pool);
//     cudaFreeHost(h_length_pool_pinned);
//     cudaFree(d_out_pool);
//     cudaFreeHost(h_out_pool_pinned);
    
//     for (int i = 0; i < nStreams; ++i) {
//         cudaStreamDestroy(streams[i]);
//         cudaEventDestroy(events[i]);
//     }


// }


/* Old synchronous version

void GpuLCS::computeLCSLengths(
		CSequence* ref,
		CSequence** sequences,
		int n_seqs,
		uint32_t* out_vector,
		CLCSBP& lcsbp)
{

    // 1. bitmasks are already in ref
    const int BATCH_SIZE = 5000;
    auto pref = ref;
    int ref_len = pref->length;
    int bv_len = (pref->data_size + 63) / 64; // number of 64-bit words needed for bitmask
    const symbol_t* ref_ptr = pref->data;

    std::vector<uint64_t> h_ref_bitmasks((size_t)(NO_SYMBOLS) * bv_len);

    for (int j = 0; j < NO_SYMBOLS * bv_len; ++j) {
        h_ref_bitmasks[j] = pref->p_bit_masks[j];
    }

    uint64_t* d_ref_bitmasks;
    cudaError_t err = cudaMalloc(&d_ref_bitmasks, h_ref_bitmasks.size() * sizeof(uint64_t));    
    err = cudaMemcpy(d_ref_bitmasks, h_ref_bitmasks.data(), h_ref_bitmasks.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        return;
    }

    uint64_t* d_workspace;
    err = cudaMalloc(&d_workspace, bv_len * sizeof(uint64_t)); // workspace for bit parallel computation
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        return;
    }

    // 2. Divide sequences into batches and send to GPU
    for (int start = 0; start < n_seqs; start += BATCH_SIZE) {
     
        int batch_n = std::min(BATCH_SIZE, n_seqs - start); // batch is smmaller at the end
        // need a vector of the sequeneces
        // need the sequences and their offsets (easier to parallelize than just lengths)

        std::vector<symbol_t> h_concat_seqs; // concatenated sequences for the batch
        std::vector<int> h_offsets(batch_n);
        std::vector<int> h_lengths(batch_n);
        int running_offset = 0;
        for (int i = 0; i < batch_n; ++i) {
            auto pseq = sequences[start + i];
            int len = pseq->length;
            h_lengths[i] = len;
            h_offsets[i] = running_offset;
            h_concat_seqs.insert(h_concat_seqs.end(), pseq->data, pseq->data + len);
            running_offset += len;
        }

        // allocate gpu mem
        symbol_t* d_concat_seqs;
        symbol_t* d_ref;
        int* d_offsets;
        int* d_lengths;
        uint32_t* d_out_lcs; // output lcs lengths

        err = cudaMalloc(&d_concat_seqs, h_concat_seqs.size() * sizeof(symbol_t));
        err = cudaMalloc(&d_ref, ref_len * sizeof(symbol_t));
        err = cudaMalloc(&d_offsets, batch_n * sizeof(int));
        err = cudaMalloc(&d_lengths, batch_n * sizeof(int));
        err = cudaMalloc(&d_out_lcs, batch_n * sizeof(uint32_t));
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
            return;
        }
        
        // copy data to GPU
        err = cudaMemcpy(d_concat_seqs, h_concat_seqs.data(), h_concat_seqs.size() * sizeof(symbol_t), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_ref, ref_ptr, ref_len * sizeof(symbol_t), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_offsets, h_offsets.data(), batch_n * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_lengths, h_lengths.data(), batch_n * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err)) ;
            return;
        }

        // 3. do kernel
        int numBlocks = 1;
        int blockSize = 1;
        LCS_Kernel<<<numBlocks, blockSize>>>(d_concat_seqs, d_ref, d_ref_bitmasks, d_workspace, bv_len, d_offsets, d_lengths, d_out_lcs, batch_n);

        // error check
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
            return;
        }

        cudaDeviceSynchronize();

        // 4. gpu to host

        std::vector<uint32_t> h_out_lcs(batch_n);
        err = cudaMemcpy(h_out_lcs.data(), d_out_lcs, batch_n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
            return;
        }

        // apply transform

        for (int i = 0; i < batch_n; ++i) {
            out_vector[start + i] = h_out_lcs[i]; // transform happens outside
        }

        // 5. free gpu mem
        cudaFree(d_concat_seqs);
        cudaFree(d_ref);
        
        cudaFree(d_offsets);
        cudaFree(d_lengths);
        cudaFree(d_out_lcs);
    
    }

    cudaFree(d_ref_bitmasks);
    cudaFree(d_workspace);


}


*/