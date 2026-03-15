# GPU-Accelerated LCS Kernel for Sequence Alignment for FAMSA2

## Content

The majority of this repository is cloned from FAMSA2. Most of our work sits inside folder `src/gpu/`. The CUDA and the CUDA header files are where our work is actually sitting.

## Usage:
Clone the entire repository with the test folder intact.

You can run `run-command.sh` to check the LCS DP matrix operation in FAMSA2.

In `run-command.sh`, uncomment the command for the tests you want to run for the CPU & GPU version in pairs on different datasets.

`GATA, helicase_C, RING, adeno_fibe` all run by picking the Thread Per Sequence kernel

`ABC_tran and transketolase_PC` will pick the Warp Per Sequence kernel

Use `diff pid.csv pid_cuda.csv` to confirm correctness

Change flag `-t` from 1 to other numbers to test multithreading.

If you really want to change the block configurations, locate the file in `src/gpu/GPU_LCS.CU`. Look for lines 541-555 for kernel launch config.
