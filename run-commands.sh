#!/bin/sh

# Change directory (DO NOT CHANGE!)
repoDir=$(dirname "$(realpath "$0")")
echo $repoDir
cd $repoDir


gmake
./bin/famsa -dist_export -pid -square_matrix ./test/adeno_fiber/adeno_fiber pid.csv
