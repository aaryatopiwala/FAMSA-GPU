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
    

    
	void computeLCSLengths(
		CSequence* ref,
		CSequence** sequences,
		int n_seqs,
		uint32_t* out_vector,
		CLCSBP& lcsbp);


template <class seq_type, class distance_type, typename Transform>
void GPUcalculateDistanceVector(
		Transform& transform,
		seq_type& ref,
		seq_type* sequences,
		int n_seqs,
		distance_type* out_vector,
		CLCSBP& lcsbp)
{

        // host-side: call the GPU worker to get raw LCS lengths,
        // then apply transform on the host to produce distances.

        // allocate host array for lcs lengths
        std::vector<uint32_t> h_lcs(n_seqs);

        // call non-template GPU worker (implemented in GPU_LCS.cu)
        computeLCSLengths(seq_to_ptr(ref), (CSequence**)sequences, n_seqs, h_lcs.data(), lcsbp);

        // apply transform locally
        auto ref_len = seq_to_ptr(ref)->length;
        for (int i = 0; i < n_seqs; ++i) {
            out_vector[i] = transform(h_lcs[i], (uint32_t)ref_len, (uint32_t)seq_to_ptr(sequences[i])->length);
        }
    
}

};