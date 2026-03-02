#include <stdint.h>
#include <vector>
#include <string>
#include <assert.h>

#include "../lcs/lcsbp.h"
#include "../utils/utils.h"

#undef min
#undef max
#include <cmath>
#include <type_traits>
#include <algorithm>
#include <stdio.h>
#include <cstring>
#include <fstream>
#include <string>
#include <iostream>



struct GpuLCS {
    

    
    template <class seq_type, class distance_type, typename Transform>
	void GPUcalculateDistanceVector(
		Transform& transform,
		seq_type& ref,
		seq_type* sequences,
		int n_seqs,
		distance_type* out_vector,
		CLCSBP& lcsbp);
};

template <class seq_type, class distance_type, typename Transform>
void GPUcalculateDistanceVector(
		Transform& transform,
		seq_type& ref,
		seq_type* sequences,
		int n_seqs,
		distance_type* out_vector,
		CLCSBP& lcsbp)
{

    // 1. bitmasks are already in ref
    const int BATCH_SIZE = 5000;
    auto pref = seq_to_ptr(ref);
    int ref_len = pref->length;
    int bv_len = (ref_len + 63) / 64; // number of 64-bit words needed for bitmask
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
            auto pseq = seq_to_ptr(sequences[start + i]);
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
            out_vector[start + i] = transform(h_out_lcs[i], ref_len, h_lengths[i]);
        }

        // 5. free gpu mem
        cudaFree(d_concat_seqs);
        cudaFree(d_ref);
        cudaFree(d_ref_bitmasks);
        cudaFree(d_offsets);
        cudaFree(d_lengths);
        cudaFree(d_out_lcs);
    
    }

        // gpu function here

}