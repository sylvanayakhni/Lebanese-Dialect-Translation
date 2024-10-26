#!/bin/bash

#SBATCH --job-name=syy06-aya-finetuning # Job name
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

MODEL_NAME="./aya-23-8b"

# --do_train \
WANDB_PROJECT=Aya_finetuning_sft

echo $WANDB_PROJECT

/apps/sw/miniconda/envs/ai-4/bin/python train_sft.py \
--dataset_name='datasets/Grammar_data.csv' \
--save_steps=100 \
--report_to='wandb' \
--logging_steps=1 \
--model_name=$MODEL_NAME \
--run_name='Aya_sft_grammar_hint_10epochs' \
--per_device_train_batch_size=16 \
--num_train_epochs=3 \
--gradient_accumulation_steps=4 \
--max_seq_length=150 \
--learning_rate=2e-4 \
--per_device_val_batch_size=16 \
--gradient_checkpointing= True \
echo 'ending' 
