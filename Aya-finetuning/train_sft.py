from pathlib import Path

import torch
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer

from args_parser import get_args
from utils import *

if __name__ == '__main__':
    args = get_args()
    experiment_name = args.run_name

    for arg in vars(args):
        print(arg, getattr(args, arg))

    print('Loading dataset')    
    dataset = get_datasets(args.dataset_name)
    #dataset = dataset.train_test_split(test_size=0.01)    
    model_name = args.model_name

    ## Bits and Bytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    torch.cuda.empty_cache()

    #Printing CUDA devices 
    device = torch.cuda.current_device()
    print("CUDA devices: ",torch.cuda.device_count())

# Get the number of GPUs
    num_gpus = torch.cuda.device_count()

# Iterate through each GPU and print its details
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  CUDA capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Current memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        print(f"  Current memory cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB\n")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        #device_map='auto',
    )

# Iterate through each GPU and print its details
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  CUDA capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Current memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        print(f"  Current memory cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB\n")


    ## Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()


    ## Print the number of trainable parameters
    print_trainable_parameters(model)

    ## Silence the warnings
    model.config.use_cache = False

    ## Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'


    ## get training arguments
    num_train_epochs= args.num_train_epochs, 
    output_dir = args.output_dir
    per_device_train_batch_size =  args.per_device_train_batch_size
    per_device_val_batch_size = args.per_device_val_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps

    ## calculate the number of steps per epoch
    epoch_steps = len(dataset) // (per_device_train_batch_size * gradient_accumulation_steps)
    print("Steps: ", epoch_steps)
    optim = args.optim
    save_steps = args.save_steps
    logging_steps = args.logging_steps
    learning_rate = args.learning_rate
    max_grad_norm = args.max_grad_norm
    warmup_ratio = args.warmup_ratio
    lr_scheduler_type = args.lr_scheduler_type



    ## Create logging and output directories
    output_dir = args.output_dir + "/"+ experiment_name
    logging_dir = args.logging_dir + "/"+ experiment_name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(logging_dir).mkdir(parents=True, exist_ok=True)
    print(f'Saving the model to {output_dir}')

    ## Create training arguments
    training_arguments = TrainingArguments(
        num_train_epochs=10,
	output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        #per_device_eval_batch_size=per_device_val_batch_size,
        #evaluation_strategy=args.evaluation_strategy,
        do_train=args.do_train,
        #do_eval=args.do_eval,
        #eval_steps=args.eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        report_to=args.report_to,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_dir=logging_dir,
        run_name = args.run_name,
    )


    ### LoRA Config
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_r = args.lora_r
    lora_target_modules = args.lora_target_modules
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules = lora_target_modules
    )


    max_seq_length = args.max_seq_length


    ## Create the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        #eval_dataset=dataset['test'],
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        formatting_func=instruction_format_hint,
    )




    ## Resume training from a checkpoint
    if args.checkpoint_path:
        trainer.train(resume_from_checkpoint=args.checkpoint_path)
    else:
        trainer.train()

    
    #Save the trained lora model
    print("Saving Lora Adapters")
    save_directory = args.saving_dir + "/"+ experiment_name
    trainer.model.save_pretrained(save_directory)


    print('Done training')
    print(trainer.model)
