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
		// start time
		auto start_time = std::chrono::high_resolution_clock::now();
        // call non-template GPU worker (implemented in GPU_LCS.cu)
        computeLCSLengths(seq_to_ptr(ref), (CSequence**)sequences, n_seqs, h_lcs.data(), lcsbp);
		// end time, print duration
		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
		std::cout << "GPU computation time: " << duration.count() << " ms" << std::endl;
        // apply transform locally
        auto ref_len = seq_to_ptr(ref)->length;
        for (int i = 0; i < n_seqs; ++i) {
            out_vector[i] = transform(h_lcs[i], (uint32_t)ref_len, (uint32_t)seq_to_ptr(sequences[i])->length);
        }
		
		//end time 2 print duration
		auto end_time_2 = std::chrono::high_resolution_clock::now();
		auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_2 - end_time);
		std::cout << "Host transform time: " << duration_2.count() << " ms" << std::endl;
    
}

};