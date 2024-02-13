#!/bin/bash

# Parameters
CONF=0

if [[ $CONF == 0 ]]; then
  N_DATA_LIST=(200 500 1000 2000 5000 10000)
  LR_LIST=(0.001)
  EPOCH_LIST=(50)
  SEED_LIST=$(seq 2000 2100)
  HORIZON_LIST=(100)
  DATA_MODEL_LIST=('AR')
  NOISE_PROFILE_LIST=('static' 'dynamic') 
  NDIM_LIST=(1)
  NOISE_LEVEL_LIST=(10)
  DELTA_LIST=(0.1)
  DELTA_TEST_LIST=(0.1)

elif [[ $CONF == 1 ]]; then
  N_DATA_LIST=(2000)
  LR_LIST=(0.001)
  EPOCH_LIST=(50)
  SEED_LIST=$(seq 2000 2100)
  HORIZON_LIST=(5 15 25 50)
  DATA_MODEL_LIST=('AR')
  NOISE_PROFILE_LIST=('static' 'dynamic')
  NDIM_LIST=(1)
  NOISE_LEVEL_LIST=(10)
  DELTA_LIST=(0.1)
  DELTA_TEST_LIST=(0.1)

# elif [[ $CONF == 2 ]]; then
#   N_DATA_LIST=(2000)
#   LR_LIST=(0.001)
#   EPOCH_LIST=(50)
#   SEED_LIST=$(seq 2000 2100)
#   HORIZON_LIST=(100)
#   DATA_MODEL_LIST=('AR+Randompeaks' 'AR+Seasonality' 'AR+VolClust')
#   NOISE_PROFILE_LIST=('static' 'dynamic')
#   NDIM_LIST=(1)
#   NOISE_LEVEL_LIST=(10)
#   DELTA_LIST=(0.1)

elif [[ $CONF == 3 ]]; then
  N_DATA_LIST=(2000)
  LR_LIST=(0.001)
  EPOCH_LIST=(50)
  SEED_LIST=$(seq 2000 2100)
  HORIZON_LIST=(100)
  DATA_MODEL_LIST=('AR')
  NOISE_PROFILE_LIST=('static' 'dynamic')
  NDIM_LIST=(2 3 5 10)
  NOISE_LEVEL_LIST=(10)
  DELTA_LIST=(0.1)
  DELTA_TEST_LIST=(0.1)

elif [[ $CONF == 4 ]]; then
  N_DATA_LIST=(2000)
  LR_LIST=(0.001)
  EPOCH_LIST=(50)
  SEED_LIST=$(seq 2000 2100)
  HORIZON_LIST=(100)
  DATA_MODEL_LIST=('AR')
  NOISE_PROFILE_LIST=('static' 'dynamic')
  NDIM_LIST=(1)
  NOISE_LEVEL_LIST=(10)
  DELTA_LIST=(0.1 0.2 0.5)
  DELTA_TEST_LIST=(1) #modified to keep consistent with delta in ts_sim.py

elif [[ $CONF == 5 ]]; then
  N_DATA_LIST=(2000)
  LR_LIST=(0.001)
  EPOCH_LIST=(50)
  SEED_LIST=$(seq 2000 2100)
  HORIZON_LIST=(100)
  DATA_MODEL_LIST=('AR')
  NOISE_PROFILE_LIST=('static' 'dynamic')
  NDIM_LIST=(1)
  NOISE_LEVEL_LIST=(10 20 50 100)
  DELTA_LIST=(0.1)
  DELTA_TEST_LIST=(0.1)

elif [[ $CONF == 6 ]]; then
  N_DATA_LIST=(2000)
  LR_LIST=(0.001)
  EPOCH_LIST=(50)
  SEED_LIST=$(seq 2000 2100)
  HORIZON_LIST=(100)
  DATA_MODEL_LIST=('AR')
  NOISE_PROFILE_LIST=('static' 'dynamic')
  NDIM_LIST=(1)
  NOISE_LEVEL_LIST=(10)
  DELTA_LIST=(0.1)
  DELTA_TEST_LIST=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
  

fi 

# Slurm parameters
MEMO=50G                             # Memory required 
TIME=00-05:00:00                    # Time required
CORE=1                              # Cores required

# Assemble order prefix #--mem="$MEMO" 
ORDP="sbatch --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Create directory for log files
LOGS="logs/simulated_data"
mkdir -p $LOGS

OUT_DIR="results/simulated_data"
mkdir -p $OUT_DIR

# Loop over configurations and chromosomes
for SEED in $SEED_LIST; do
  for N_DATA in "${N_DATA_LIST[@]}"; do
    for LR in "${LR_LIST[@]}"; do
      for EPOCH in "${EPOCH_LIST[@]}"; do
        for HORIZON in "${HORIZON_LIST[@]}"; do
          for DATA_MODEL in "${DATA_MODEL_LIST[@]}"; do
            for NOISE_PROFILE in "${NOISE_PROFILE_LIST[@]}"; do
              for NDIM in "${NDIM_LIST[@]}"; do
                for NOISE_LEVEL in "${NOISE_LEVEL_LIST[@]}"; do
                  for DELTA in "${DELTA_LIST[@]}"; do
                    for DELTA_TEST in "${DELTA_TEST_LIST[@]}"; do
                      JOBN="n"$N_DATA"_h"$HORIZON"_datamodel"$DATA_MODEL"_noise"$NOISE_PROFILE"_seed"$SEED"_ndim"$NDIM"_level"$NOISE_LEVEL"_delta"$DELTA"_deltat"$DELTA_TEST
                      OUT_FILE=$OUT_DIR"/"$JOBN".txt"
                      COMPLETE=0
                      #ls $OUT_FILE
                      if [[ -f $OUT_FILE ]]; then
                      COMPLETE=1
                      fi
                      if [[ $COMPLETE -eq 0 ]]; then
                      # Script to be run
                      SCRIPT="ts_sim.sh ${N_DATA} ${LR} ${EPOCH} ${SEED} ${HORIZON} ${DATA_MODEL} ${NOISE_PROFILE} ${NDIM} ${NOISE_LEVEL} ${DELTA} ${DELTA_TEST}"
                      echo $SCRIPT
                      # Define job name for this chromosome
                      OUTF=$LOGS"/"$JOBN".out"
                      ERRF=$LOGS"/"$JOBN".err"
                      # Assemble slurm order for this job
                      ORD="${ORDP} -J ${JOBN} -o ${OUTF} -e ${ERRF} ${SCRIPT}"
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
        done
      done
    done
  done
done
