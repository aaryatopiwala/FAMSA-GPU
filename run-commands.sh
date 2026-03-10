#!/bin/sh

repoDir=$(dirname "$(realpath "$0")")
cd "$repoDir"

# time ./bin/famsa -t 1 -dist_export -pid -square_matrix ./test/GATA/GATA pid.csv
# time ./bin/famsa -t 1 -cuda -dist_export -pid -square_matrix ./test/GATA/GATA pid_cuda.csv
# time ./bin/famsa -t 1 -dist_export -pid -square_matrix ./test/RING/RING pid.csv
time ./bin/famsa -t 1 -cuda -dist_export -pid -square_matrix ./test/RING/RING pid_cuda.csv

# LOG=run_output.txt
# DIFF=diff.txt

# echo "===== FAMSA Benchmark =====" > "$LOG"
# echo "Directory: $repoDir" >> "$LOG"
# echo "" >> "$LOG"

# echo "Compiling..." | tee -a "$LOG"
# gmake >> "$LOG" 2>&1

# echo "" | tee -a "$LOG"
# echo "===== CPU Run =====" | tee -a "$LOG"
# # (time ./bin/famsa -dist_export -pid -square_matrix ./test/GATA/GATA pid.csv) >> "$LOG" 2>&1
# # (time ./bin/famsa -dist_export -pid -square_matrix ./test/transketolase_PC/transketolase_PC_n pid.csv)
# # rm pid.csv

# (time ./bin/famsa -dist_export -pid -square_matrix ./test/transketolase_PC/transketolase_PC_n STDOUT > /dev/null) >> "$LOG" 2>&1
# echo "" | tee -a "$LOG"
# echo "===== GPU Run =====" | tee -a "$LOG"
# # (time ./bin/famsa -cuda -dist_export -pid -square_matrix ./test/GATA/GATA pid_cuda.csv) >> "$LOG" 2>&1
# # (time ./bin/famsa -cuda -dist_export -pid -square_matrix ./test/transketolase_PC/transketolase_PC_n pid_cuda.csv)
# # rm pid_cuda.csv

# (time ./bin/famsa -cuda -dist_export -pid -square_matrix ./test/transketolase_PC/transketolase_PC_n STDOUT > /dev/null) >> "$LOG" 2>&1
# echo "" | tee -a "$LOG"
# echo "===== Done =====" | tee -a "$LOG"
# echo "Results saved to $LOG"