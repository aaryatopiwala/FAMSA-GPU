
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
#include <chrono>

// Persistent GPU/host pools (allocate once, reuse across calls)
static bool g_pools_inited = false;
static int g_nStreams_default = 4;

static symbol_t* g_d_concat_pool = nullptr;
static symbol_t* g_h_concat_pool_pinned = nullptr;
static size_t g_stream_concat_len_cap = 0; // per-stream element capacity

static int* g_d_offset_pool = nullptr;
static int* g_h_offset_pool_pinned = nullptr;
static size_t g_stream_offset_len_cap = 0;

static int* g_d_length_pool = nullptr;
static int* g_h_length_pool_pinned = nullptr;
static size_t g_stream_length_len_cap = 0;

static uint32_t* g_d_out_pool = nullptr;
static uint32_t* g_h_out_pool_pinned = nullptr;
static size_t g_stream_output_len_cap = 0; // per-stream element capacity (uint32_t)

static uint64_t* g_d_ref_bitmasks = nullptr;
static size_t g_d_ref_bitmasks_bytes = 0;

static uint64_t* g_d_workspace = nullptr;
static size_t g_workspace_bv_len_cap = 0; // bv_len per stream

static uint64_t* g_d_carry = nullptr;
static int g_carry_num_steps_cap = 0; // num_steps per stream

static void printCudaError(const char* msg, cudaError_t err){
    fprintf(stderr, "GPU_ERROR %s: %s (%s)\n", msg, cudaGetErrorString(err), cudaGetErrorName(err));
}

static bool ensureGpuPools(int nStreams,
                          size_t stream_concat_len_elems,
                          size_t stream_offset_len_elems,
                          size_t stream_output_len_elems,
                          size_t bv_len,
                          int num_steps)
{
    cudaError_t err;
    if (!g_pools_inited) {
        g_nStreams_default = nStreams;
        g_pools_inited = true;
    }

    // concat pool (per-stream capacity)
    if (stream_concat_len_elems > g_stream_concat_len_cap) {
        if (g_d_concat_pool) cudaFree(g_d_concat_pool);
        if (g_h_concat_pool_pinned) cudaFreeHost(g_h_concat_pool_pinned);
        size_t total_elems = stream_concat_len_elems * (size_t)g_nStreams_default;
        err = cudaMalloc(&g_d_concat_pool, total_elems * sizeof(symbol_t));
        if (err != cudaSuccess) { printCudaError("cudaMalloc d_concat_pool", err); return false; }
        err = cudaMallocHost(&g_h_concat_pool_pinned, total_elems * sizeof(symbol_t));
        if (err != cudaSuccess) { printCudaError("cudaMallocHost h_concat_pool_pinned", err); return false; }
        g_stream_concat_len_cap = stream_concat_len_elems;
    }

    // offsets
    if (stream_offset_len_elems > g_stream_offset_len_cap) {
        if (g_d_offset_pool) cudaFree(g_d_offset_pool);
        if (g_h_offset_pool_pinned) cudaFreeHost(g_h_offset_pool_pinned);
        size_t total = stream_offset_len_elems * (size_t)g_nStreams_default;
        err = cudaMalloc(&g_d_offset_pool, total * sizeof(int));
        if (err != cudaSuccess) { printCudaError("cudaMalloc d_offset_pool", err); return false; }
        err = cudaMallocHost(&g_h_offset_pool_pinned, total * sizeof(int));
        if (err != cudaSuccess) { printCudaError("cudaMallocHost h_offset_pool_pinned", err); return false; }
        g_stream_offset_len_cap = stream_offset_len_elems;
    }

    // lengths
    if (stream_offset_len_elems > g_stream_length_len_cap) {
        if (g_d_length_pool) cudaFree(g_d_length_pool);
        if (g_h_length_pool_pinned) cudaFreeHost(g_h_length_pool_pinned);
        size_t total = stream_offset_len_elems * (size_t)g_nStreams_default;
        err = cudaMalloc(&g_d_length_pool, total * sizeof(int));
        if (err != cudaSuccess) { printCudaError("cudaMalloc d_length_pool", err); return false; }
        err = cudaMallocHost(&g_h_length_pool_pinned, total * sizeof(int));
        if (err != cudaSuccess) { printCudaError("cudaMallocHost h_length_pool_pinned", err); return false; }
        g_stream_length_len_cap = stream_offset_len_elems;
    }

    // output
    if (stream_output_len_elems > g_stream_output_len_cap) {
        if (g_d_out_pool) cudaFree(g_d_out_pool);
        if (g_h_out_pool_pinned) cudaFreeHost(g_h_out_pool_pinned);
        size_t total = stream_output_len_elems * (size_t)g_nStreams_default;
        err = cudaMalloc(&g_d_out_pool, total * sizeof(uint32_t));
        if (err != cudaSuccess) { printCudaError("cudaMalloc d_out_pool", err); return false; }
        err = cudaMallocHost(&g_h_out_pool_pinned, total * sizeof(uint32_t));
        if (err != cudaSuccess) { printCudaError("cudaMallocHost h_out_pool_pinned", err); return false; }
        g_stream_output_len_cap = stream_output_len_elems;
    }

    // ref bitmasks
    size_t ref_bytes = (size_t)NO_SYMBOLS * bv_len * sizeof(uint64_t);
    if (ref_bytes > g_d_ref_bitmasks_bytes) {
        if (g_d_ref_bitmasks) cudaFree(g_d_ref_bitmasks);
        err = cudaMalloc(&g_d_ref_bitmasks, ref_bytes);
        if (err != cudaSuccess) { printCudaError("cudaMalloc d_ref_bitmasks", err); return false; }
        g_d_ref_bitmasks_bytes = ref_bytes;
    }

    // workspace per stream (bv_len * nStreams)
    if (bv_len > g_workspace_bv_len_cap) {
        if (g_d_workspace) cudaFree(g_d_workspace);
        size_t total = bv_len * (size_t)g_nStreams_default;
        err = cudaMalloc(&g_d_workspace, total * sizeof(uint64_t));
        if (err != cudaSuccess) { printCudaError("cudaMalloc d_workspace", err); return false; }
        g_workspace_bv_len_cap = bv_len;
    }

    // carry pool (num_steps per stream)
    if (num_steps > g_carry_num_steps_cap) {
        if (g_d_carry) cudaFree(g_d_carry);
        size_t total = (size_t)num_steps * g_nStreams_default;
        err = cudaMalloc(&g_d_carry, total * sizeof(uint64_t));
        if (err != cudaSuccess) { printCudaError("cudaMalloc d_carry", err); return false; }
        g_carry_num_steps_cap = num_steps;
    }

    return true;
}


__global__ void LCS_Kernel(
    const symbol_t* d_concat_seqs,
    //const symbol_t* d_ref,
    const uint64_t* d_ref_bitmasks,
    uint64_t* d_carry,
    uint64_t* d_workspace,
    int bv_len,
    int chunk_size,
    int iteration_step_size,
    const int* d_offsets,
    const int* d_lengths,
    uint32_t* d_out_lcs,
    int batch_n)
{
    int gs = gridDim.x;
    int bs = blockDim.x;
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    extern __shared__ uint64_t shared_mem[];
    uint64_t* s_ref_bitmasks = shared_mem;                          // shared memory for reference bitmasks, size will be NO_SYMBOLS * bv_len
    uint64_t* s_workspace    = shared_mem + NO_SYMBOLS * bv_len;    // shared memory workspace for bit vectors of the current row
    uint64_t* s_carry        = s_workspace + bv_len;                // shared memory for carry bits for the current wavefront step

    // Load reference bitmasks into shared memory for faster access for each block
    if (tx < NO_SYMBOLS * bv_len) {
        for (int i = tx; i < NO_SYMBOLS * bv_len; i+=bs) {
            s_ref_bitmasks[i] = d_ref_bitmasks[i];
        }   
    }
    __syncthreads(); // Wait for reference bitmasks to be loaded into shared memory

    // serial method that will be parallelized later
    // if (tx == 0 && bx == 0){
    //     for (int i = 0; i < batch_n; ++i) {
    //         const symbol_t* seq = d_concat_seqs + d_offsets[i];
    //         int seq_len = d_lengths[i];
    //         int ref_len = 0; // need to pass this in or compute from d_ref
            
    //         for (int w = 0; w < bv_len; ++w) {
    //             s_workspace[w] = ~((uint64_t)0); // initialize workspace
    //         }

    //         for (int j = 0; j < seq_len; ++j) {
    //             unsigned char c = seq[j];
    //             if ( c == 22 || c >= 32) { // unknown symbol, or over symbol range, skip
    //                 continue;
    //             }
    //             const uint64_t* s0b = s_ref_bitmasks + c * bv_len; // bitmask for symbol c
    //             uint64_t carry = 0;
    //             for (int w = 0; w < bv_len; ++w) {
    //                 uint64_t V = s_workspace[w];
    //                 uint64_t tb = V & s0b[w];
    //                 uint64_t V2 = V + tb + carry;
    //                 carry = (V2 < V) ? 1 : 0; // detect overflow
    //                 s_workspace[w] = V2 | (V - tb);
    //             }
    //         }
            
    //         uint32_t res = 0;
    //         for (int w = 0; w < bv_len; ++w) {
    //             res += __popcll(~s_workspace[w]); // count set bits in ~workspace
    //         }
    //         d_out_lcs[i] = res; 
    //     }
    // }

    // if (bx == 0){
        // for (int i = 0; i < batch_n; ++i) {
        for (int b = bx; b < batch_n; b += gs) {
            const symbol_t* seq = d_concat_seqs + d_offsets[b];
            int seq_len = d_lengths[b];
            int ref_len = 0; // need to pass this in or compute from d_ref
            
            if (tx < bv_len) {
                for (int w = tx; w < bv_len; w+=bs) {
                    s_workspace[w] = ~((uint64_t)0); // initialize workspace, set bit vector of each chunk to all 1s
                }
            }
            __syncthreads();

            int num_steps = (seq_len + iteration_step_size - 1) / iteration_step_size;
            if (tx < num_steps){
                for (int n = tx; n < bv_len * num_steps; n += bs){
                    s_carry[n] = 0;
                }
            }
            __syncthreads(); // Wait for the whole row to be loaded

           for (int k = 0; k < (bv_len + num_steps - 1); ++k) {
                int i_min = max(0, k - (num_steps - 1));
                int i_max = min(bv_len - 1, k);

                for (int row = i_min + tx; row <= i_max; row += bs) {
                    int i = row;
                    int j = k - row;

                    uint64_t carry = s_carry[i * num_steps + j];
                    unsigned char c = seq[j];
                    if (c != 22 && c < 32) {
                        const uint64_t* s0b = s_ref_bitmasks + c * bv_len;
                        uint64_t V = s_workspace[i];
                        uint64_t tb = V & s0b[i];
                        uint64_t sum1 = V + tb;
                        uint64_t V2 = sum1 + carry;
                        s_workspace[i] = V2 | (V - tb);
                        carry = (sum1 < V) || (V2 < sum1);
                    }
                    if (i + 1 < bv_len)
                        s_carry[(i + 1) * num_steps + j] = carry;
                }
                __syncthreads();
            }

            if (tx == 0) {
                uint32_t res = 0;
                for (int w = 0; w < bv_len; ++w) {
                    res += __popcll(~s_workspace[w]);
                }
                d_out_lcs[b] = res;
            }
        __syncthreads();
        }

    // }
}

    
void GpuLCS::computeLCSLengths(
		CSequence* ref,
		CSequence** sequences,
		int n_seqs,
		uint32_t* out_vector,
		CLCSBP& lcsbp)
{

    // 1. bitmasks are already in ref
    // start time
    auto start_time = std::chrono::high_resolution_clock::now();
    const int BATCH_SIZE = 4000;
    auto pref = ref;
    int ref_len = pref->length;
    int chunk_size = 64;
    int bv_len = (pref->data_size + chunk_size - 1) / chunk_size; // number of chunks needed for bitmask
    int iteration_step_size = 1; // number of characters each thread will process in the inner loop to amortize overhead. Tune this for best performance
    const symbol_t* ref_ptr = pref->data;


    // find the longest possible length among batches to use for concat, also max sequence length
    int max_batch_len = 0;
    int max_seq_len = 0;
    for (int start = 0; start < n_seqs; start += BATCH_SIZE) {
        size_t cur = 0;
        int end = min(start + BATCH_SIZE, n_seqs);
        for (int i = start; i < end; ++i) {
            cur += sequences[i]->length;
            max_seq_len = max(max_seq_len, sequences[i]->length);
        }
        max_batch_len = max(max_batch_len, (int)cur);
    }
    size_t stream_concat_len = max_batch_len;
    size_t stream_offset_len = BATCH_SIZE;
    size_t stream_output_len = BATCH_SIZE * sizeof(uint32_t);

    std::vector<uint64_t> h_ref_bitmasks((size_t)(NO_SYMBOLS) * bv_len);

    for (int j = 0; j < NO_SYMBOLS * bv_len; ++j) {
        h_ref_bitmasks[j] = pref->p_bit_masks[j];
    }

    // We'll copy h_ref_bitmasks into the persistent device pool after ensuring pools are sized
    cudaError_t err = cudaSuccess;

    int nStreams = 4;
    std::vector<cudaStream_t> streams(nStreams);
    std::vector<cudaEvent_t> events(nStreams);
    std::vector<cudaEvent_t> h2d_start(nStreams);
    std::vector<cudaEvent_t> h2d_stop(nStreams);
    std::vector<cudaEvent_t> kernel_start(nStreams);
    std::vector<cudaEvent_t> kernel_stop(nStreams);
    std::vector<cudaEvent_t> d2h_start(nStreams);
    std::vector<cudaEvent_t> d2h_stop(nStreams);
    for (int i = 0; i < nStreams; ++i) {
        err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
            return;
        }
        err = cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
            return;
        }
        err = cudaEventCreate(&h2d_start[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
            return;
        }
        err = cudaEventCreate(&h2d_stop[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
            return;
        }
        err = cudaEventCreate(&kernel_start[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
            return;
        }
        err = cudaEventCreate(&kernel_stop[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
            return;
        }
        err = cudaEventCreate(&d2h_start[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
            return;
        }
        err = cudaEventCreate(&d2h_stop[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
            return;
        }
    }

    int num_steps = (max_seq_len + iteration_step_size - 1) / iteration_step_size; // number of steps needed to process the longest sequence in the batch

    // Ensure persistent pools exist and are large enough. stream_output_len is bytes; convert to uint32_t element count
    size_t stream_output_len_elems = stream_output_len / sizeof(uint32_t);
    bool ok = ensureGpuPools(nStreams, stream_concat_len, stream_offset_len, stream_output_len_elems, bv_len, num_steps);
    if (!ok) {
        fprintf(stderr, "GPU_ERROR: ensureGpuPools failed\n");
        return;
    }

    // Copy reference bitmasks into persistent device buffer (only necessary when ref changes)
    size_t ref_bytes = (size_t)NO_SYMBOLS * bv_len * sizeof(uint64_t);
    err = cudaMemcpy(g_d_ref_bitmasks, h_ref_bitmasks.data(), ref_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy d_ref_bitmasks", err); return; }

    std::vector<bool> stream_busy(nStreams, false); // track which streams are busy
    std::vector<int> stream_batch_start(nStreams, 0); // track which batch is in which stream
    std::vector<int> stream_batch_n(nStreams, 0); // track batch size for each stream

    //end time before batching starts
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "GPU setup time: " << duration.count() << " ms" << std::endl;
    // 2. Divide sequences into batches and send to GPU
    for (int start = 0; start < n_seqs; start += BATCH_SIZE) {
     
        int batch_n = std::min(BATCH_SIZE, n_seqs - start); // batch is smmaller at the end
        int batch_idx = start / BATCH_SIZE;
        int stream_idx = batch_idx % nStreams;
        cudaStream_t currStream = streams[stream_idx];
        uint64_t* currCarry = g_d_carry + stream_idx * num_steps; // carry for current stream
        uint64_t* currWorkspace = g_d_workspace + stream_idx * bv_len; // workspace for current stream
        // need a vector of the sequeneces
        // need the sequences and their offsets (easier to parallelize than just lengths)

        std::vector<symbol_t> h_concat_seqs; // concatenated sequences for the batch
        std::vector<int> h_offsets(batch_n);
        std::vector<int> h_lengths(batch_n);
        h_concat_seqs.reserve(batch_n * max_seq_len); // reserve enough space to avoid reallocations
        int running_offset = 0;
        for (int i = 0; i < batch_n; ++i) {
            auto pseq = sequences[start + i];
            int len = pseq->length;
            h_lengths[i] = len;
            h_offsets[i] = running_offset;
            h_concat_seqs.insert(h_concat_seqs.end(), pseq->data, pseq->data + len);
            running_offset += len;
        }

        // per stream slice of pool
        symbol_t* d_concat_seqs = g_d_concat_pool + stream_idx * g_stream_concat_len_cap;
        symbol_t* h_concat_seqs_pinned = g_h_concat_pool_pinned + stream_idx * g_stream_concat_len_cap;
        int* d_offsets = g_d_offset_pool + stream_idx * g_stream_offset_len_cap;
        int* h_offsets_pinned = g_h_offset_pool_pinned + stream_idx * g_stream_offset_len_cap;
        int* d_lengths = g_d_length_pool + stream_idx * g_stream_length_len_cap;
        int* h_lengths_pinned = g_h_length_pool_pinned + stream_idx * g_stream_length_len_cap;
        uint32_t* d_out_lcs = g_d_out_pool + stream_idx * g_stream_output_len_cap;
        uint32_t* h_out_lcs_pinned = g_h_out_pool_pinned + stream_idx * g_stream_output_len_cap;
        
        // check if stream is busy

        if (stream_busy[stream_idx]) {
            // wait for stream to finish
            cudaEventSynchronize(events[stream_idx]);
            // copy results back to output vector since async D2H is done
            int prev_start = stream_batch_start[stream_idx];
            int prev_n = stream_batch_n[stream_idx];
            for (int i = 0; i < prev_n; ++i) {
                out_vector[prev_start + i] = g_h_out_pool_pinned[stream_idx * stream_output_len / sizeof(uint32_t) + i];
            }
            stream_busy[stream_idx] = false;
            float ms_h2d = 0, ms_kernel = 0, ms_d2h = 0;
            cudaEventElapsedTime(&ms_h2d, h2d_start[stream_idx], h2d_stop[stream_idx]);
            cudaEventElapsedTime(&ms_kernel, kernel_start[stream_idx], kernel_stop[stream_idx]);
            cudaEventElapsedTime(&ms_d2h, d2h_start[stream_idx], d2h_stop[stream_idx]);
            printf("Stream %d batch %d: H2D=%.2f ms, Kernel=%.2f ms, D2H=%.2f ms\n", stream_idx, stream_batch_start[stream_idx] / BATCH_SIZE, ms_h2d, ms_kernel, ms_d2h);
        }

        size_t concat_size = h_concat_seqs.size() * sizeof(symbol_t);
        memcpy(h_concat_seqs_pinned, h_concat_seqs.data(), concat_size);
        memcpy(h_offsets_pinned, h_offsets.data(), batch_n * sizeof(int));
        memcpy(h_lengths_pinned, h_lengths.data(), batch_n * sizeof(int));

        // copy data to GPU asynchronously
        cudaEventRecord(h2d_start[stream_idx], currStream);
        err = cudaMemcpyAsync(d_concat_seqs, h_concat_seqs_pinned, h_concat_seqs.size() * sizeof(symbol_t), cudaMemcpyHostToDevice, currStream);
        //err = cudaMemcpyAsync(d_ref, h_ref_pinned, ref_len * sizeof(symbol_t), cudaMemcpyHostToDevice, currStream);
        err = cudaMemcpyAsync(d_offsets, h_offsets_pinned, batch_n * sizeof(int), cudaMemcpyHostToDevice, currStream);
        err = cudaMemcpyAsync(d_lengths, h_lengths_pinned, batch_n * sizeof(int), cudaMemcpyHostToDevice, currStream);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err)) ;
            return;
        }
        cudaEventRecord(h2d_stop[stream_idx], currStream);

        // 3. do kernel
        int numBlocks = 1024;
        int blockSize = 256;
        
        
        // asynch
        cudaEventRecord(kernel_start[stream_idx], currStream);
        size_t smem_bytes = (NO_SYMBOLS * bv_len + bv_len + bv_len * num_steps) * sizeof(uint64_t);
        LCS_Kernel<<<numBlocks, blockSize, smem_bytes, currStream>>>(d_concat_seqs, g_d_ref_bitmasks, currCarry, currWorkspace, bv_len, chunk_size, iteration_step_size, d_offsets, d_lengths, d_out_lcs, batch_n);
        cudaEventRecord(kernel_stop[stream_idx], currStream);

        // error check
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
            return;
        }


        // 4. gpu to host

        cudaEventRecord(d2h_start[stream_idx], currStream);
        err = cudaMemcpyAsync(h_out_lcs_pinned, d_out_lcs, batch_n * sizeof(uint32_t), cudaMemcpyDeviceToHost, currStream);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
            return;
        }
        cudaEventRecord(d2h_stop[stream_idx], currStream);

        cudaEventRecord(events[stream_idx], currStream);
        stream_busy[stream_idx] = true;
        stream_batch_start[stream_idx] = start;
        stream_batch_n[stream_idx] = batch_n;

    
    }

    // drain remaining streams
    for (int i = 0; i < nStreams; ++i) {
        if (stream_busy[i]) {
            cudaEventSynchronize(events[i]);
            int prev_start = stream_batch_start[i];
            int prev_n = stream_batch_n[i];
            uint32_t* host_slice = g_h_out_pool_pinned + i * stream_output_len / sizeof(uint32_t);
            //for (int j = 0; j < prev_n; ++j) {
            //    out_vector[prev_start + j] = host_slice[j];
            //}
            memcpy(out_vector + prev_start, host_slice, prev_n * sizeof(uint32_t));
            float ms_h2d = 0, ms_kernel = 0, ms_d2h = 0;
            cudaEventElapsedTime(&ms_h2d, h2d_start[i], h2d_stop[i]);
            cudaEventElapsedTime(&ms_kernel, kernel_start[i], kernel_stop[i]);
            cudaEventElapsedTime(&ms_d2h, d2h_start[i], d2h_stop[i]);
            printf("Stream %d batch %d: H2D=%.2f ms, Kernel=%.2f ms, D2H=%.2f ms\n", i, stream_batch_start[i] / BATCH_SIZE, ms_h2d, ms_kernel, ms_d2h);
        }
    }

    // persistent pools are kept allocated across calls to reuse memory and avoid allocation overhead
    // (do not free g_* pools here)
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    for (int i = 0; i < nStreams; ++i) {
        cudaEventDestroy(h2d_start[i]);
        cudaEventDestroy(h2d_stop[i]);
        cudaEventDestroy(kernel_start[i]);
        cudaEventDestroy(kernel_stop[i]);
        cudaEventDestroy(d2h_start[i]);
        cudaEventDestroy(d2h_stop[i]);
    }

    // end of batching time
    auto end_time2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - end_time);
    std::cout << "GPU batching and computation time: " << duration2.count() << " ms" << std::endl;

}


 // Old synchronous version

/*
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
        //symbol_t* d_ref;
        int* d_offsets;
        int* d_lengths;
        uint32_t* d_out_lcs; // output lcs lengths
        uint64_t* d_carry; 

        err = cudaMalloc(&d_concat_seqs, h_concat_seqs.size() * sizeof(symbol_t));
        //err = cudaMalloc(&d_ref, ref_len * sizeof(symbol_t));
        err = cudaMalloc(&d_offsets, batch_n * sizeof(int));
        err = cudaMalloc(&d_lengths, batch_n * sizeof(int));
        err = cudaMalloc(&d_out_lcs, batch_n * sizeof(uint32_t));
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
            return;
        }
        
        // copy data to GPU
        err = cudaMemcpy(d_concat_seqs, h_concat_seqs.data(), h_concat_seqs.size() * sizeof(symbol_t), cudaMemcpyHostToDevice);
        //err = cudaMemcpy(d_ref, ref_ptr, ref_len * sizeof(symbol_t), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_offsets, h_offsets.data(), batch_n * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_lengths, h_lengths.data(), batch_n * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err)) ;
            return;
        }

        int iteration_step_size = 1;
        int chunk_size = 64;
        int max_seq_len = 0;
        for (int i = 0; i < batch_n; ++i) {
            max_seq_len = std::max(max_seq_len, h_lengths[i]);
        }
        int num_steps = (max_seq_len + iteration_step_size - 1) / iteration_step_size;
        err = cudaMalloc(&d_carry, num_steps * sizeof(uint64_t));
        err = cudaMemset(d_carry, 0, num_steps * sizeof(uint64_t)); // initialize carry to 0
        // 3. do kernel
        int numBlocks = 1024;
        int blockSize = 256;
        size_t smem_bytes = (NO_SYMBOLS * bv_len + bv_len + bv_len * num_steps) * sizeof(uint64_t);
        LCS_Kernel<<<numBlocks, blockSize, smem_bytes>>>(d_concat_seqs, g_d_ref_bitmasks, d_carry, d_workspace, bv_len, chunk_size, iteration_step_size, d_offsets, d_lengths, d_out_lcs, batch_n);

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
        //cudaFree(d_ref);
        cudaFree(d_carry);
        
        cudaFree(d_offsets);
        cudaFree(d_lengths);
        cudaFree(d_out_lcs);
    
    }

    cudaFree(d_ref_bitmasks);
    cudaFree(d_workspace);


}

*/