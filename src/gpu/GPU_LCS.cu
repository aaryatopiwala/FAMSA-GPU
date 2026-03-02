
#include "GPU_LCS.cuh"
#include <vector>
#include <stdint.h>
#include <algorithm>
#include <stdio.h>
#include <cstring>
#include <fstream>
#include <string>
#include <iostream>


__global__ void LCS_Kernel(
    const symbol_t* d_concat_seqs,
    const symbol_t* d_ref,
    const uint64_t* d_ref_bitmasks,
    uint64_t* d_workspace,
    int bv_len,
    const int* d_offsets,
    const int* d_lengths,
    uint32_t* d_out_lcs,
    int batch_n)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // serial method that will be parallelized later
    if (tx == 0 && bx == 0){
        for (int i = 0; i < batch_n; ++i) {
            const symbol_t* seq = d_concat_seqs + d_offsets[i];
            int seq_len = d_lengths[i];
            int ref_len = 0; // need to pass this in or compute from d_ref
            
            for (int w = 0; w < bv_len; ++w) {
                d_workspace[w] = ~((uint64_t)0); // initialize workspace
            }

            for (int j = 0; j < seq_len; ++j) {
                unsigned char c = seq[j];
                if (c >= NO_SYMBOLS) {
                    continue;
                }
                const uint64_t* s0b = d_ref_bitmasks + c * bv_len; // bitmask for symbol c
                uint64_t carry = 0;
                for (int w = 0; w < bv_len; ++w) {
                    uint64_t V = d_workspace[w];
                    uint64_t tb = V & s0b[w];
                    uint64_t V2 = V + tb + carry;
                    carry = (V2 < V) ? 1 : 0; // detect overflow
                    d_workspace[w] = V2 | (V - tb);
                }
            }
            
            uint32_t res = 0;
            for (int w = 0; w < bv_len; ++w) {
                res += __popcll(~d_workspace[w]); // count set bits in ~workspace
            }
            d_out_lcs[i] = res; 
        }
    }
}

