#!/bin/bash
#SBATCH --job-name=gan_train
#SBATCH --output=gan_train_%j.out
#SBATCH --error=gan_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=shard:24  
#SBATCH --cpus-per-task=4       
#SBATCH --time=48:00:00
#SBATCH --mem=32G

module load anaconda3-2024.2
module load cuda-12.8
source ~/.bashrc
conda activate gan


cd /home/tanmoyhazra/gan_project
python ganvaelpips.py
