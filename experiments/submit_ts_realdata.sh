#!/bin/bash

# Parameters
CONF=0

if [[ $CONF == 0 ]]; then
  N_DATA_LIST=(200 500 1000)
  LR_LIST=(0.001)
  EPOCH_LIST=(50)
  SEED_LIST=$(seq 2000 2100)
  NOISE_PROFILE_LIST=('static' 'dynamic')
  NOISE_LEVEL_LIST=(10)
  DELTA_LIST=(0.1)

elif [[ $CONF == 1 ]]; then
  N_DATA_LIST=(1000)
  LR_LIST=(0.001)
  EPOCH_LIST=(50)
  SEED_LIST=$(seq 2000 2100)
  NOISE_PROFILE_LIST=('static' 'dynamic')
  NOISE_LEVEL_LIST=(5 10 20 50)
  DELTA_LIST=(0.1)

fi 

# Slurm parameters
MEMO=10G                          
TIME=00-03:00:00                   
CORE=1                              

# Assemble order prefix #--mem="$MEMO" 
ORDP="sbatch --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Create directory for log files
LOGS="logs/real_data"
mkdir -p $LOGS

OUT_DIR="results/real_data"
mkdir -p $OUT_DIR

# Loop over configurations and chromosomes
for SEED in $SEED_LIST; do
  for N_DATA in "${N_DATA_LIST[@]}"; do
    for LR in "${LR_LIST[@]}"; do
      for EPOCH in "${EPOCH_LIST[@]}"; do
        for NOISE_PROFILE in "${NOISE_PROFILE_LIST[@]}"; do
          for NOISE_LEVEL in "${NOISE_LEVEL_LIST[@]}"; do
            JOBN="n"$N_DATA"_noise"$NOISE_PROFILE"_level"$NOISE_LEVEL"_seed"$SEED
            OUT_FILE=$OUT_DIR"/"$JOBN".txt"
            COMPLETE=0
            #ls $OUT_FILE
            if [[ -f $OUT_FILE ]]; then
            COMPLETE=1
            fi
            if [[ $COMPLETE -eq 0 ]]; then
            # Script to be run
            SCRIPT="ts_realdata.sh $N_DATA $LR $EPOCH $SEED $NOISE_PROFILE $NOISE_LEVEL"
            # Define job name for this chromosome
            OUTF=$LOGS"/"$JOBN".out"
            ERRF=$LOGS"/"$JOBN".err"
            # Assemble slurm order for this job
            ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
            # Print order
            echo $ORD
            # Submit order
            $ORD
            fi
          done 
        done
      done
    done
  done
done
