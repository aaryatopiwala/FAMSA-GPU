# GPU-Accelerated LCS Kernel for Sequence Alignment for FAMSA2

## Usage:
clone the repository with test folder intact.

You can run run-command.sh to check the LCS DP matrix operation in FAMSA2.

In run-command.sh, uncomment tests you want to run for CPU or GPU version on different datasets.

`GATA, helicase_C, RING, adeno_fibe` all runs by picking Thread Per Sequence kernel

`ABC_tran and transketolase_PC` will pick Warp Per Sequence kernel

Use `diff pid.csv pid_cuda.csv` to confirm correctness

Change flag `-t` from 1 to other numbers to test multithreading.
