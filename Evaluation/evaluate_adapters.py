import torch
import os
import gc
import pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import argparse
from comet import load_from_checkpoint
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument('--checkpoint_path', type=str, default='../lora_adapters/Aya_cpo_LWSS')   
    parser.add_argument('--data_name', type=str, default='../datasets/Baladi_sentences.csv')
    parser.add_argument('--output_dir', type=str, default="./outputs/Baladi")
    args, _ = parser.parse_known_args()
    return args


def generate_prompt(question, examples, translations, bad_translations, rare_words):
    # The initial part of the prompt
    prompt_start = '''### Instruction: \nYou are a skilled translator with expertise in Lebanese colloquial language. It is very important to focus on correctly translating the following words {rare_words}

Examples:
'''
    prompt_start = prompt_start.replace('rare_words', ', '.join(rare_words))
    # Generate the examples section
    examples_section = ''
    for ex, trans, bad_trans in zip(examples, translations,bad_translations):
        examples_section += f'\n\n###Input:{ex}\n###Hint: We prefer to translate it to \n###Response:\n{trans} \nrather than\n{bad_trans}'

    # Combine all parts of the prompt
    full_prompt = prompt_start + examples_section + f'\n\n###Input:{{Question}}\n###Hint: We prefer to translate it to \n###Response:\n'

    # Replace the {Question} placeholder with the actual question
    full_prompt = full_prompt.replace('{Question}', question)
    
    message= [{"role": "user", "content": full_prompt}]

    return message

def get_message_format_contrastive(text):
    prompt = "### Instruction: \nTranslate the following sentences from Lebanese Arabic to English.\n\n###Examples:\n\n ### Input: اليوم رح نحكي عن قصة مثل لبناني معروف ومعظمنا سامع فيه بس ما منعرف شو قصتو.\n ### Hint: We prefer to translate it to\n### Response:\nToday, we will speak about the story of a renowned Lebanese proverb, which most of us have heard of, but we do not know the story behind it.\nrather than\nToday we will talk about a story of a famous Lebanese man and most of us have heard of him but do not know his story.\n\n ### Input: وكتير انبسطت لما شفتا، لأنو حسيت إني عضو مهِم من مجتمع بحب الفن والشعر ودايما بجرب يساهم بحضورو.\n ### Hint: We prefer to translate it to\n### Response:\nI was very happy when I saw it, because I felt that I am an important member of a society that loves poetry and art and always tries to contribute by attending.\nrather than \nI was very happy to see you, because I felt like a member of a community that I love, art and poetry, and I always try to contribute by attending.\n\n ### Input: {Question}\n ### Hint: We prefer to translate it to\n### Response:\n"
    message = prompt.format_map({'Question':text})
    message= [{"role": "user", "content": message}]
    return message

def get_message_format_few_shot(text):
    prompt = "### Instruction: \nTranslate the following sentences from Lebanese Arabic to English.\n\n###Examples:\n\n ### Input: اليوم رح نحكي عن قصة مثل لبناني معروف ومعظمنا سامع فيه بس ما منعرف شو قصتو.\n### Response:Today, we will speak about the story of a renowned Lebanese proverb, which most of us have heard of, but we do not know the story behind it.\n\n ### Input: وكتير انبسطت لما شفتا، لأنو حسيت إني عضو مهِم من مجتمع بحب الفن والشعر ودايما بجرب يساهم بحضورو.\n### Response:I was very happy when I saw it, because I felt that I am an important member of a society that loves poetry and art and always tries to contribute by attending.\n\n ### Input: {Question}\n### Response:"
    message = prompt.format_map({'Question':text})
    message= [{"role": "user", "content": message}]
    return message    


def get_message_format(text):
    prompt = "Translate from Lebanese Arabic to English: {Question}"
    message = prompt.format_map({'Question':text})
    message= [{"role": "user", "content": message}]
    return message


def get_response(text,tokenizer,model,device):

    prompt = get_message_format_contrastive(text)
    #prompt = get_message_format(text)
    #prompt = get_message_format_few_shot(text)
    input_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding= True)
    inputs = input_ids.to(device)

    #prompt_padded_len = len(input_ids[0]
    gen_tokens = model.generate(
        inputs,
        max_new_tokens=400, 
        do_sample=True, 
        temperature=0.4,
)

    response = tokenizer.decode(
    gen_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
)

    response = response.split("<|CHATBOT_TOKEN|>")[-1].strip()
    response =  response.split("rather than")[0].strip()
    return response

if __name__ == '__main__':
    xcomet_path= "./models--Unbabel--XCOMET-XL/snapshots/baa17625e541fe87c4c0010616e35eab12c864f7/checkpoints/model.ckpt"
    args = parse_args()
    checkpoint_path  = args.checkpoint_path
    model_name = "../aya-23-8b"
    data_name = args.data_name
    output_dir = args.output_dir
   
    print(data_name)
    print(output_dir) 
    
    with open(data_name, 'r', encoding='utf-8', errors='ignore') as file:
        data = pd.read_csv(file)
    
    data_apc = data["apc_Arabic"].tolist() 
    print(len(data_apc))
    
      ## Bits and Bytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        #device_map="auto",
        #quantization_config=bnb_config,
        device_map={"": 0},
    )
    
    print("Successfully loaded the model into memory")
    # Merge base model with the adapter
   # model = PeftModel.from_pretrained(model, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"2. Successfully loaded peft model into memory")


    generations = []
    for sentence in data_apc: 
        generations.append(get_response(sentence,tokenizer=tokenizer, model=model, device=device))    
    print("Successfully generated outputs")
    
    #Save the outputs to the output directory
    output_dir = output_dir + f"/{checkpoint_path.split('/')[-2]}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    translations = pd.DataFrame(generations, columns=["translations"])
    saving_dir = output_dir + "/generations.csv" 
    translations.to_csv(saving_dir) 
    print("Successfully saved outputs to: ", saving_dir)
    
    
    #Evaluation
    print("Evaluation Started!")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    xcomet_model = load_from_checkpoint(xcomet_path)
    print("XComet model loaded")


    if data_name=="../datasets/Baladi_sentences.csv":
        predictions = generations
        sources= data_apc
        
        eval_data = []
        for src, mt in zip(sources, predictions):
            eval_data.append({"src": src, "mt": mt})

    else:
        predictions = generations[:369]
        sources= data_apc[:369]
        references = data["English"].tolist()[:369]

        eval_data = []
        for src, mt, ref in zip(sources, predictions, references):
            eval_data.append({"src": src, "mt": mt, "ref": ref})
        
    xcomet_data = xcomet_model.predict(eval_data, batch_size=32, gpus=1)
    
    print("Evaluation was successful")
    print ("Xcomet_score", xcomet_data.system_score)
    
    xcomet_dir = output_dir + "/evaluation.txt"

    with open(xcomet_dir, 'a') as f: 
         f.write('Xcomet_score: ' + str(xcomet_data.system_score) + '\n')
         f.write('Xcomet_scores: '+ str(xcomet_data.scores) + '\n')
         f.write('Xcomet_error_spans: ' + str(xcomet_data.metadata.error_spans) + '\n')
         f.flush()


    print("Evaluation scores saved Successfully!")
       
