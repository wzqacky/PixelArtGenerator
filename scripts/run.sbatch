#!/bin/bash
#SBATCH --job-name=pixel

#SBATCH --time=30:00:00
#SBATCH --mail-user=zqwu@connect.ust.hk
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --nodes=1 --gres=gpu:1

source ~/.bashrc
conda init
conda activate pixel
bash scripts/train_palette_controlnet_sdxl.sh