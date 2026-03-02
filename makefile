all: famsa

# *** REFRESH makefile utils
include refresh.mk

$(call INIT_SUBMODULES)
$(call INIT_GLOBALS)
$(call CHECK_OS_ARCH, $(PLATFORM))

# *** Project directories
$(call SET_SRC_OBJ_BIN,src,obj,bin)
3RD_PARTY_DIR := ./libs

SRC_SIMD_DIR := $(SRC_DIR)/simd
OBJ_SIMD_DIR := $(OBJ_DIR)/simd

# *** Project configuration
$(call ADD_MIMALLOC, $(3RD_PARTY_DIR)/mimalloc)
$(call PROPOSE_ZLIB_NG, $(3RD_PARTY_DIR)/zlib-ng)
$(call PROPOSE_ISAL, $(3RD_PARTY_DIR)/isa-l)
$(call CHOOSE_GZIP_DECOMPRESSION)
$(call ADD_LIBDEFLATE, $(3RD_PARTY_DIR)/libdeflate)
$(call ADD_REFRESH_LIB, $(3RD_PARTY_DIR))
$(call SET_STATIC, $(STATIC_LINK))
$(call SET_C_CPP_STANDARDS, c11, c++20)
$(call SET_GIT_COMMIT)

$(call SET_FLAGS, $(TYPE))

$(call SET_COMPILER_VERSION_ALLOWED, GCC, Linux_x86_64, 11, 20)
$(call SET_COMPILER_VERSION_ALLOWED, GCC, Linux_aarch64, 11, 20)
$(call SET_COMPILER_VERSION_ALLOWED, GCC, Darwin_x86_64, 11, 13)
$(call SET_COMPILER_VERSION_ALLOWED, GCC, Darwin_arm64, 11, 13)

ifneq ($(MAKECMDGOALS),clean)
$(call CHECK_COMPILER_VERSION)
endif

# *** Source files and rules
$(eval $(call PREPARE_DEFAULT_COMPILE_RULE,MAIN,))
$(eval $(call PREPARE_DEFAULT_COMPILE_RULE,CORE,core))
$(eval $(call PREPARE_DEFAULT_COMPILE_RULE,LCS,lcs))
$(eval $(call PREPARE_DEFAULT_COMPILE_RULE,TREE,tree))
$(eval $(call PREPARE_DEFAULT_COMPILE_RULE,UTILS,utils))

# *** SIMD rules
# Main kmer-db files
ifeq ($(ARCH_TYPE),x86_64)
SRC_SIMD := $(SRC_SIMD_DIR)/lcsbp_avx_intr.cpp $(SRC_SIMD_DIR)/lcsbp_avx2_intr.cpp $(SRC_SIMD_DIR)/lcsbp_avx512_intr.cpp $(SRC_SIMD_DIR)/utils_avx.cpp $(SRC_SIMD_DIR)/utils_avx2.cpp 
$(OBJ_SIMD_DIR)/lcsbp_avx_intr.cpp.o: $(SRC_SIMD_DIR)/lcsbp_avx_intr.cpp
	@mkdir -p $(OBJ_SIMD_DIR)
	$(CXX) $(CPP_FLAGS_AVX) $(OPTIMIZATION_FLAGS) $(ARCH_FLAGS) $(INCLUDE_DIRS) -MMD -MF $@.d -c $< -o $@
$(OBJ_SIMD_DIR)/lcsbp_avx2_intr.cpp.o: $(SRC_SIMD_DIR)/lcsbp_avx2_intr.cpp
	@mkdir -p $(OBJ_SIMD_DIR)
	$(CXX) $(CPP_FLAGS_AVX2) $(OPTIMIZATION_FLAGS) $(ARCH_FLAGS) $(INCLUDE_DIRS) -MMD -MF $@.d -c $< -o $@
$(OBJ_SIMD_DIR)/lcsbp_avx512_intr.cpp.o: $(SRC_SIMD_DIR)/lcsbp_avx512_intr.cpp
	@mkdir -p $(OBJ_SIMD_DIR)
	$(CXX) $(CPP_FLAGS_AVX512) $(OPTIMIZATION_FLAGS) $(ARCH_FLAGS) $(INCLUDE_DIRS) -MMD -MF $@.d -c $< -o $@
$(OBJ_SIMD_DIR)/utils_avx.cpp.o: $(SRC_SIMD_DIR)/utils_avx.cpp
	@mkdir -p $(OBJ_SIMD_DIR)
	$(CXX) $(CPP_FLAGS_AVX) $(OPTIMIZATION_FLAGS) $(ARCH_FLAGS) $(INCLUDE_DIRS) -MMD -MF $@.d -c $< -o $@
$(OBJ_SIMD_DIR)/utils_avx2.cpp.o: $(SRC_SIMD_DIR)/utils_avx2.cpp
	@mkdir -p $(OBJ_SIMD_DIR)
	$(CXX) $(CPP_FLAGS_AVX2) $(OPTIMIZATION_FLAGS) $(ARCH_FLAGS) $(INCLUDE_DIRS) -MMD -MF $@.d -c $< -o $@
else
SRC_SIMD := $(SRC_SIMD_DIR)/lcsbp_neon_intr.cpp $(SRC_SIMD_DIR)/utils_neon.cpp 
$(OBJ_SIMD_DIR)/lcsbp_neon_intr.cpp.o: $(SRC_SIMD_DIR)/lcsbp_neon_intr.cpp
	@mkdir -p $(OBJ_SIMD_DIR)
	$(CXX) $(CPP_FLAGS_NEON) $(OPTIMIZATION_FLAGS) $(ARCH_FLAGS) $(INCLUDE_DIRS) -MMD -MF $@.d -c $< -o $@
$(OBJ_SIMD_DIR)/utils_neon.cpp.o: $(SRC_SIMD_DIR)/utils_neon.cpp
	@mkdir -p $(OBJ_SIMD_DIR)
	$(CXX) $(CPP_FLAGS_NEON) $(OPTIMIZATION_FLAGS) $(ARCH_FLAGS) $(INCLUDE_DIRS) -MMD -MF $@.d -c $< -o $@
endif

OBJ_SIMD := $(patsubst $(SRC_SIMD_DIR)/%.cpp, $(OBJ_SIMD_DIR)/%.cpp.o, $(SRC_SIMD))

# Dependency files (needed only for SIMD)
-include $(OBJ_SIMD:.o=.o.d)

# stuff for cuda
CUDA_HOME ?= /usr/local/cuda
NVCC := $(shell command -v nvcc 2>/dev/null)

# set nvcc path and CUDA_HOME based on env vars
ifeq ($(NVCC),)
  ifdef CUDA_HOME
    NVCC := $(CUDA_HOME)/bin/nvcc
  else
    CUDA_HOME ?= /usr/local/cuda
    NVCC := $(CUDA_HOME)/bin/nvcc
  endif
else
  # get cuda from nvcc path
  CUDA_HOME := $(patsubst %/bin/nvcc,%,$(NVCC))
endif

CUDA_INC := -I$(CUDA_HOME)/include
CUDA_LIBS := -L$(CUDA_HOME)/lib64 -lcudart

# cpp 12, sm80 is a30
CUFLAGS := -std=c++20 -O3 -Xcompiler -fPIC -arch=sm_80 -rdc=true --use_fast_math

GPU_SRCS := $(SRC_DIR)/gpu/GPU_LCS.cu
GPU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cu.o,$(GPU_SRCS))

$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CUFLAGS) $(CUDA_INC) -c $< -o $@


famsa: $(OUT_BIN_DIR)/famsa
$(OUT_BIN_DIR)/famsa: mimalloc_obj \
	$(OBJ_MAIN) $(OBJ_CORE) $(OBJ_LCS) $(OBJ_TREE) $(OBJ_UTILS) $(OBJ_SIMD) $(GPU_OBJS)
	-mkdir -p $(OUT_BIN_DIR)	
	$(NVCC) -o $@  \
	$(MIMALLOC_OBJ) \
	$(OBJ_MAIN) $(OBJ_CORE) $(OBJ_LCS) $(OBJ_TREE) $(OBJ_UTILS) $(OBJ_SIMD) $(GPU_OBJS) \
	$(LIBRARY_FILES) $(filter-out -fabi-version=6,$(LINKER_FLAGS)) $(LINKER_DIRS) $(CUDA_LIBS)

# *** Cleaning
.PHONY: clean init
clean: clean-libdeflate clean-mimalloc_obj clean-zlib-ng clean-isa-l
	-rm -r $(OBJ_DIR)
	-rm -r $(OUT_BIN_DIR)

init:
	$(call INIT_SUBMODULES)
