#!/bin/bash -l
#SBATCH -t 48:00:00                  # walltime
#SBATCH -N 1                         # one node
#SBATCH -n 48                         # CPU (hyperthreaded) cores
#SBATCH -J myo_sweep   # job name
#SBATCH --mem=48GB                   # memory per node in GB
#SBATCH -o myo_sweep.log
#SBATCH -e myo_sweep.err
#SBATCH --gres=gpu:4
#SBATCH --constraint=10GB      
#SBATCH --mail-type=END
#SBATCH --mail-user="federicoclaudi@protonmail.com"

# ---------------------------------- module ---------------------------------- #

echo "Loading modules"
source /etc/profile.d/modules.sh
module use /cm/shared/modulefiles

# ----------------------------------- task ----------------------------------- #

echo "Running python script"

# cd to project
cd /om2/user/claudif/DecodingAlgorithms/BCI_ALVI_challenge_competition

# execute in singularity container
conda activate test2

python sweep.py
