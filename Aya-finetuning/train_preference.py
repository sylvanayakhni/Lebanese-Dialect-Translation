from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import DPOTrainer,ORPOTrainer, ORPOConfig, CPOTrainer, CPOConfig

from args_parser import get_args
from utils import *
from accelerate import PartialState

if __name__ == '__main__':
    args = get_args()
 

    for arg in vars(args):
        print(arg, getattr(args, arg))

    print('Loading dataset', args.dataset_name)
    dataset = get_datasets(dataset_name= args.dataset_name)
    test_dataset= get_datasets('datasets/test_data.csv')
    
    ##dataset = dataset.select(range(31700)) 
    original_columns = dataset.column_names
    test_original_columns = test_dataset.column_names
    
    train_dataset = dataset.map(dpo_format,remove_columns=original_columns)
    test_dataset = test_dataset.map(dpo_format_test,remove_columns=test_original_columns)
    
    print(train_dataset[0])
    print(test_dataset[0])

    
    run_name = args.run_name
    model_name = args.model_name

    # Clearing CUDA cache 
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

    ## Bits and Bytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    #device_string = PartialState().process_index
    #device_map = {'':device_string}

    ## Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": 0} ,
        #device_map=device_map,
        #device_map='auto'
    )

    if args.checkpoint_path:
        print('checkpoint captured')
        model.enable_input_require_grads()
        model = PeftModel.from_pretrained(model, args.checkpoint_path,is_trainable=True)
        #model._mark_only_adapters_as_trainable()



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
    max_seq_length = args.max_seq_length 
        
    ## calculate the number of steps per epoch
    epoch_steps = len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)
    print("Steps: ", epoch_steps)
    optim = args.optim
    save_steps = args.save_steps
    logging_steps = args.logging_steps
    learning_rate = args.learning_rate
    max_grad_norm = args.max_grad_norm
    warmup_ratio = args.warmup_ratio
    lr_scheduler_type = args.lr_scheduler_type
    


    ## Create logging and output d:irectories
    output_dir = args.output_dir + "/" + run_name
    loggig_dir = args.logging_dir +  "/" + run_name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(loggig_dir).mkdir(parents=True, exist_ok=True)
    print(f'Saving the model to {output_dir}')
    
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


    if args.finetuning_technique=='dpo':
    ## Create training arguments
        training_arguments = TrainingArguments(
            num_train_epochs=3,
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            per_device_eval_batch_size=per_device_val_batch_size,
            evaluation_strategy=args.evaluation_strategy,
            do_train=args.do_train,
            do_eval=args.do_eval,
            eval_steps=args.eval_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=5e-5,
            fp16=True,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            report_to=args.report_to,
            gradient_checkpointing=args.gradient_checkpointing,
            logging_dir=loggig_dir,
    )     
	 
    
    ## Create the trainer
        trainer = DPOTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            max_length=max_seq_length,
            tokenizer=tokenizer,
	        beta = 0.1,
            args=training_arguments,
    )


    elif args.finetuning_technique=='orpo':
        orpo_args = ORPOConfig(
            num_train_epochs=3,
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
	    beta = 0.1,
            #per_device_eval_batch_size=per_device_val_batch_size,
            #evaluation_strategy=args.evaluation_strategy,
            do_train=args.do_train,
            #do_eval=args.do_eval,
            #eval_steps=args.eval_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=8e-6,
            fp16=True,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            report_to=args.report_to,
            gradient_checkpointing=args.gradient_checkpointing,
            logging_dir=loggig_dir,
            max_length= 512,
     )
    ## Create the trainer
        trainer = ORPOTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            tokenizer=tokenizer,
            args=orpo_args,
    )

    elif args.finetuning_technique=='cpo':
    ## Create training arguments
        cpo_args = CPOConfig(
            #num_train_epochs=3,
            max_steps=args.max_steps,
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            per_device_eval_batch_size=per_device_val_batch_size,
            evaluation_strategy=args.evaluation_strategy,
            do_train=args.do_train,
            do_eval=args.do_eval,
            eval_steps=args.eval_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=args.learning_rate,
            fp16=True,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            report_to=args.report_to,
            gradient_checkpointing=args.gradient_checkpointing,
            #gradient_checkpointing_kwargs= {"use_reentrant": False},
            logging_dir=loggig_dir, 
            #cpo_alpha = ,
            max_length= max_seq_length,
            max_prompt_length= 128,
    )

    ## Create the trainer
        trainer = CPOTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=peft_config,
            tokenizer=tokenizer,
            args=cpo_args,
    )
        
    else: 
        print("No finetuning technique is specified")
    
    

    ## Resume training from a checkpoint
    if args.checkpoint_path:
        trainer.train(resume_from_checkpoint=args.checkpoint_path)
    else:
        trainer.train()

    
    #Save the trained lora model
    print("Saving Lora Adapters")
    save_directory = args.saving_dir +  "/" + run_name
    trainer.model.save_pretrained(save_directory)


    print('Done training')
    print(trainer.model)
