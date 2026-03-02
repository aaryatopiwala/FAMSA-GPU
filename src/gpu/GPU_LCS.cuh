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