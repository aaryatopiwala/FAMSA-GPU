#!/bin/sh

repoDir=$(dirname "$(realpath "$0")")
cd $repoDir

LOG=run_output.txt

echo "===== FAMSA Benchmark =====" > $LOG
echo "Directory: $repoDir" >> $LOG
echo "" >> $LOG

echo "Compiling..." | tee -a $LOG
gmake >> $LOG 2>&1

echo "" | tee -a $LOG
echo "===== CPU Run =====" | tee -a $LOG

(time ./bin/famsa -dist_export -pid -square_matrix ./test/GATA/GATA pid.csv) >> $LOG 2>&1

echo "" | tee -a $LOG
echo "===== GPU Run =====" | tee -a $LOG

(time ./bin/famsa -cuda -dist_export -pid -square_matrix ./test/GATA/GATA pid_cuda.csv) >> $LOG 2>&1

echo "" | tee -a $LOG
echo "===== Done =====" | tee -a $LOG
echo "Results saved to $LOG"