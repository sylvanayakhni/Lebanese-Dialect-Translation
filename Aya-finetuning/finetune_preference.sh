#!/bin/bash

#SBATCH --job-name=syy06-aya-finetuning_dpo # Job name
#SBATCH --partition=msfea-ai
#SBATCH --account=syy06


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --time=01-00:00:00
#SBATCH --account=aya_project

source ~/.bashrc

echo 'starting.......................'
###################### RUN LLM Finetune ######################

MODEL_NAME="./aya-23-8b"

# --do_train \
WANDB_PROJECT=Aya_cpo_LW


echo $WANDB_PROJECT

/apps/sw/miniconda/envs/ai-4/bin/python train_preference.py \
--dataset_name='datasets/language_wave_story_data_sentences.csv' \
--finetuning_technique='cpo' \
--save_steps=50 \
--report_to='wandb' \
--logging_steps=10 \
--eval_steps=10 \
--model_name='aya-23-8b' \
--run_name='Aya_cpo_LW_after_sft_flores' \
--per_device_train_batch_size=4 \
--per_device_val_batch_size=4 \
--num_train_epochs=3 \
--gradient_accumulation_steps=2 \
--max_seq_length=256 \
--gradient_checkpointing=True \
--max_steps=200 \
--learning_rate=1e-4 \
--checkpoint_path='experiments/Aya_sft_flores/checkpoint-50' 

echo ' ending '
