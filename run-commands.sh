#!/bin/sh

repoDir=$(dirname "$(realpath "$0")")
cd "$repoDir"

gmake

# Thread Per Sequence Kernel datasets

# time ./bin/famsa -t 1 -dist_export -pid -square_matrix ./test/GATA/GATA pid.csv
# time ./bin/famsa -t 1 -cuda -dist_export -pid -square_matrix ./test/GATA/GATA pid_cuda.csv

time ./bin/famsa -t 1 -dist_export -pid -square_matrix ./test/helicase_C/helicase_C_n_80k pid.csv
time ./bin/famsa -t 1 -cuda -dist_export -pid -square_matrix ./test/helicase_C/helicase_C_n_80k pid_cuda.csv

# time ./bin/famsa -t 1 -dist_export -pid -square_matrix ./test/RING/RING pid.csv
# time ./bin/famsa -t 1 -cuda -dist_export -pid -square_matrix ./test/RING/RING pid_cuda.csv


# Warp Per Sequence Kernel

# time ./bin/famsa -t 1 -dist_export -pid -square_matrix ./test/ABC_tran/ABC_tran_n pid.csv
# time ./bin/famsa -t 1 -cuda -dist_export -pid -square_matrix ./test/ABC_tran/ABC_tran_n pid_cuda.csv

# time ./bin/famsa -t 1 -dist_export -pid -square_matrix ./test/transketolase_PC/transketolase_PC_n_80k pid.csv
# time ./bin/famsa -t 1 -cuda -dist_export -pid -square_matrix ./test/transketolase_PC/transketolase_PC_n_80k pid_cuda.csv

