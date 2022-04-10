#!/bin/bash
#SBATCH --job-name=CNN_train_gpu
#SBATCH --output=%x.o%j
#SBATCH --time=01:00:00
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Module load
module load anaconda3/2020.02/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment code
source activate music_classification

# Train the network
time scripts/train_CNN.sh -g #option enable the GPU accelaration
