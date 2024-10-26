#!/bin/bash

#SBATCH --job-name=syy06-evaluation # Job name
#SBATCH --partition=gpu
#SBATCH --account=syy06


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --time=0-06:00:00
#SBATCH --account=aya_project

source ~/.bashrc

echo 'starting.......................'
###################### RUN LLM Finetune ######################


module load python/ai-4

echo 'generation starting'

python evaluate_adapters.py \
--checkpoint_path='../experiments/Aya_sft_OSF/checkpoint-100' \
--data_name='../datasets/Baladi.csv' \
--output_dir='./outputs/Baladi_contrastive' \

echo ' ending' 


echo 'generation starting'

python evaluate_adapters.py \
--checkpoint_path='../experiments/Aya_sft_madar/checkpoint-100' \
--data_name='../datasets/Baladi.csv' \
--output_dir='./outputs/Baladi_contrastive' \

echo ' ending' 

echo 'generation starting'

python evaluate_adapters.py \
--checkpoint_path='../experiments/Aya_sft_flores_32/checkpoint-100' \
--data_name='../datasets/Baladi.csv' \
--output_dir='./outputs/Baladi_contrastive' \

echo ' ending' 

echo 'generation starting'

python evaluate_adapters.py \
--checkpoint_path='../experiments/Aya_contrastive_OSF/checkpoint-100' \
--data_name='../datasets/Baladi.csv' \
--output_dir='./outputs/Baladi_contrastive' \

echo ' ending' 

echo 'generation starting'

python evaluate_adapters.py \
--checkpoint_path='../experiments/Aya_contrastive_flores/checkpoint-100' \
--data_name='../datasets/Baladi.csv' \
--output_dir='./outputs/Baladi_contrastive' \

echo ' ending' 