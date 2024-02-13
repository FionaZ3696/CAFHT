#!/bin/bash

module purge
module load gcc/8.3.0
module load cuda/11.2.0
module load cudnn/8.1.0.77-11.2-cuda
source ~/.bashrc 
eval "$(conda shell.bash hook)"
conda activate expt

python3 ts_realdata.py $1 $2 $3 $4 $5 $6
