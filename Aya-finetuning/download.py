from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch
import bitsandbytes as bnb
from datasets import load_dataset
from trl import SFTTrainer
from datasets import Dataset
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import re
import wandb

MODEL_PATH = "inceptionai/jais-13b"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = AutoModelForCausalLM.from_pretrained(
          MODEL_PATH,
          trust_remote_code=True , 
          torch_dtype=torch.bfloat16,
          cache_dir='./jais-base',
          device_map="auto",
        )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

tokenizer.save_pretrained(save_directory)
