import re
from datasets import load_dataset
#import pyarabic.araby as araby



def print_trainable_parameters(model):
    '''
    Prints the number of trainable parameters in the model.
    '''
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}'
    )


def instruction_format_dpo(example):
    output_texts = []
    for i in range(len(example['apc_Arabic'])):
        rejected = f"{example['Jais Translations'][i]}"
        prompt = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|> Translate from Lebanese Arabic to English :{example['apc_Arabic'][i]}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        chosen = f"{example['English'][i]}"
        output_texts.append({
			"prompt": prompt,
			"chosen": chosen,
			"rejected": rejected,	
	})
    return output_texts


def instruction_format_contrastive(example):
    output_texts = []
    for i in range(len(example['apc_Arabic'])):
        text = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|><INS>Translate from Lebanese Arabic to English.<INPUT>{example['apc_Arabic'][i]} <HINT>An undesirable translation would be:{example['Aya-8b'][i]}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{example['English'][i]}"                        

        output_texts.append(text)

    return output_texts

def instruction_format(example):
    output_texts = []
    for i in range(len(example['apc_Arabic'])):
        text = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|> Translate from Lebanese Arabic to English :{example['apc_Arabic'][i]}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{example['English'][i]}"
        output_texts.append(text)
    return output_texts


def instruction_format_hint(example):
    output_texts = []
    for i in range(len(example['Lebanese'])):
        text = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|> You are a Lebanese-English translator. Translate the input sentence from Lebanese Arabic to English. When needed, a hint will be provided to guide your translation. Try to make good use of it.\nInput:{example['Lebanese'][i]}\nHint:{example['Hint'][i]}\n<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{example['English'][i]}"
        output_texts.append(text)
    return output_texts


def get_datasets(dataset_name='datasets/language_wave_story_data.csv', field='prompt'):
    '''
    Returns train, validation and test datasets for arabic, in the format described in
    generate_arabic_training_prompt().
    '''
    train_dataset = load_dataset("csv", data_files= dataset_name, split= "all")
    dataset = train_dataset.shuffle(seed=42)
    return dataset


def dpo_format(example):
    rejected = f"{example['Aya-8b']}"
    prompt = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|> Translate from Lebanese Arabic to English :{example['apc_Arabic']}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    chosen = f"{example['English']}"
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        }
def get_response(text,tokenizer,model,device):
    prompt = get_message_format(text)
    input_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding= True)
    inputs = input_ids.to(device)

    #prompt_padded_len = len(input_ids[0])

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
    return response


def get_message_format(text):
    prompt = "Translate from Lebanese Arabic to English: {Question}"
    message = prompt.format_map({'Question':text})
    message= [{"role": "user", "content": message}]
    return message	
