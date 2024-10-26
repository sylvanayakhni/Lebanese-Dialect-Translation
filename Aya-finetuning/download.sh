#!/bin/bash

#SBATCH --job-name=syy06-download # Job name
#SBATCH --partition=msfea-ai
#SBATCH --account=syy06


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --time=03-00:00:00
#SBATCH --account=aya_project

source ~/.bashrc

echo 'starting.......................'
###################### RUN LLM Finetune ######################


echo $WANDB_PROJECT


/apps/sw/miniconda/envs/ai-4/bin/python download.py 

echo ' ending '
