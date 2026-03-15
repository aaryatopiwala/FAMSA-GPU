# GPU-Accelerated LCS Kernel for Sequence Alignment for FAMSA2

## Usage:
Clone the entire repository with the test folder intact.

You can run `run-command.sh` to check the LCS DP matrix operation in FAMSA2.

In `run-command.sh`, uncomment the command for the tests you want to run for the CPU & GPU version in pairs on different datasets.

`GATA, helicase_C, RING, adeno_fibe` all run by picking the Thread Per Sequence kernel

`ABC_tran and transketolase_PC` will pick the Warp Per Sequence kernel

Use `diff pid.csv pid_cuda.csv` to confirm correctness

Change flag `-t` from 1 to other numbers to test multithreading.
