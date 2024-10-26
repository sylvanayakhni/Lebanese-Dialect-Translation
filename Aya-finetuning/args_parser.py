import argparse


def get_args():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args, _ = parser.parse_known_args()
    return args


def add_args(parser: argparse.ArgumentParser):
####################################### Model #######################################
    parser.add_argument('--finetuning_technique',
                            type=str,
                            default='dpo')
    parser.add_argument('--model_name',
                            type=str,
                            default='./aya-23-8b')
    parser.add_argument('--use_flash_attention_2',
                            type=bool,
                            default=False)
    parser.add_argument('--report_to',
                            type=str,   
                            default='wandb')
    parser.add_argument('--save_steps',
                            type=int,
                            default=1000)
    parser.add_argument('--max_seq_length',
                            type=int,
                            default=128)
    parser.add_argument('--checkpoint_path',
                            type=str,
                            default=None)
    parser.add_argument('--do_eval',
                            type=bool,
                            default=True)
    parser.add_argument('--do_train',
                            type=bool,
                            default= True)   
    parser.add_argument('--evaluation_strategy',
                            type=str,
                            default='steps')
    parser.add_argument('--eval_steps',
                            type=int,
                            default=50)
    
####################################### Logs #######################################

    parser.add_argument('--logging_dir',
                            type=str,
                            default='./logs')
    parser.add_argument('--logging_steps',
                            type=int,
                            default=10)
    parser.add_argument('--log_level',
                            type=str,
                            default='info')
    parser.add_argument('--logging_strategy',
                            type=str,
                            default='steps')
    parser.add_argument('--save_total_limit',
                            type=int,
                            default=10)
    parser.add_argument('--run_name',
                            type=str,
                            default='aya_sft_OPF')
    
 ####################################### Data #######################################
    
    parser.add_argument('--dataset_name',
                            type=str,
                            default='datasets/Open_subtitles_filtered.csv')
    parser.add_argument('--field',
                            type=str,
                            default='prompt')
    parser.add_argument('--per_device_train_batch_size',
                            type=int,
                            default=4)
    parser.add_argument('--per_device_val_batch_size',
                            type=int,
                            default=1)

####################################### Training #######################################

    parser.add_argument('--num_train_epochs',
                            type=str,
                            default=3)

    parser.add_argument('--output_dir',
                            type=str,
                            default='./experiments')
    parser.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=16)
    parser.add_argument('--optim',
                            type=str,
                            default='paged_adamw_32bit')
    parser.add_argument('--learning_rate',
                            type=float,
                            default=2e-4)
    parser.add_argument('--max_grad_norm',
                            type=float,
                            default=0.3)
    parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.03)
    parser.add_argument('--lr_scheduler_type',
                            type=str,
                            default='constant')
    parser.add_argument('--group_by_length',
                            type=bool,
                            default=True)
    
####################################### QLoRA #######################################

    parser.add_argument('--bnb_4bit_quant_type',
                            type=str,
                            default='nf4')
    parser.add_argument('--bnb_4bit_compute_dtype',
                            type=str,
                            default='bfloat16')
    parser.add_argument('--bnb_4bit_use_double_quant',
                            type=bool,
                            default=True)
    parser.add_argument('--gradient_checkpointing',
                            type=bool,
                            default=False)
    
    ##################################### LORA #####################################

    parser.add_argument('--lora_alpha',
                            type=int,
                            default=16)
    parser.add_argument('--lora_dropout',
                            type=float,
                            default=0.1)
    parser.add_argument('--lora_r',
                            type=int,
                            default=16)
    parser.add_argument('--bias',
                            type=str,
                            default='none')
    parser.add_argument('--task_type',
                            type=str,
                            default='CAUSAL_LM')
    parser.add_argument('--lora_target_modules',
                            type=str,
                            nargs='+',
                            default=[
                                "q_proj", 
                                "v_proj", 
                                "k_proj", 
                                "o_proj",
                            ])
    
    ####################################### Saving #######################################
    parser.add_argument('--saving_dir',
                            type=str,
                            default='./lora_adapters')

