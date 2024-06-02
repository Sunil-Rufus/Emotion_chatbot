import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ['TRANSFORMERS_CACHE'] = '/home/csgrad/sunilruf/'
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.cuda.empty_cache()
device = "cuda" # the device to load the model onto
#model_path = "/home/csgrad/sunilruf/emotion_chatbot/master/sunils_code/mistra/experiment-6/checkpoint-9/"
model_path = "/home/csgrad/sunilruf/emotion_chatbot/master/sunils_code/mistra/rank_64/rank_64_1/checkpoint-15/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, device_map="auto")

import json

json_file_path = '../../ExTES.json'  # Change 'output.json' to your desired file name

# Write the list to the JSON file
with open(json_file_path, 'rb') as json_file:
    extes_data = json.load(json_file)
    
file_path = '.././scenes.txt'  # Replace 'your_text_file.txt' with the actual path to your text file

# Open the file in read mode
with open(file_path, 'r') as file:
    # Read all lines from the file and split them based on newlines
    scenes = file.read().splitlines()

# Now, 'lines' is a list containing each line from the file
print(scenes)

emotion_data = []
not_available = []
for i in range(len(extes_data)):
    try:
        if (extes_data[i]['scene'].strip().lower() or extes_data[i]['description'].strip().lower()) in scenes:
            
            emotion_data.append(extes_data[i])
            
    except:
        not_available.append(i)
        
def create_list(data, value):
    s = []
    for i in range(len(data)):
        #print(value.lower())
        if data[i]['scene'].lower().rstrip() == value.lower():
            s.append(data[i])
            
    return s
    
combined_scenes = []
for i in range(0,21):
    
    s = create_list(emotion_data, scenes[i])
    if len(s)>200:
        combined_scenes.append(s)

combined_scenes[0] = combined_scenes[0][0:1000]

formatted_data1 = []

for i in range(len(combined_scenes)):
    for item in combined_scenes[i]:
        formatted_data = []
        scene = item['scene'].lower()
        #description = item['description']
        content = item['content']
        """formatted_data.append({"scene": scene, "role": "scene"})
        if item['description']:
            formatted_data.append({"description": description, "role": "description"})"""
        
        formatted_data.append({"scene": scene, "role": "scene"})
        for element in content:
            for role, content_text in element.items():
                if role=="User":
                    role = "user"
                elif role=="AI":
                    role = "assistant"
                else:
                    continue    
                    
                formatted_data.append({"role": role, "content": content_text})
                
        formatted_data1.append(formatted_data)
        
import pandas as pd

# Define column names
columns = ['scene', 'content']

# Create an empty DataFrame with specified columns
df = pd.DataFrame(columns=columns)

# Display the empty DataFrame
print(df)

for i in range(len(formatted_data1)):
    empty_dict = {}
    empty_dict['scene'] = formatted_data1[i][0]['scene']
    #empty_dict['description'] = formatted_data1[i][1]['description']  
    empty_dict['content'] = formatted_data1[i]
    
    df = df.append(empty_dict, ignore_index=True)
    
df.dropna(inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in sss.split(df, df['scene']):
    X_train, X_test = df.iloc[train_index], df.iloc[test_index]
    
X_train['content'] = X_train['content'].apply(lambda x: x[1:])
X_test['content'] = X_test['content'].apply(lambda x: x[1:])

from datasets import DatasetDict
from datasets import Dataset
datasets_train_test = DatasetDict({
    "train": Dataset.from_pandas(X_train),
    "test": Dataset.from_pandas(X_test)
    })

import ast 
def apply_chat_template(example, tokenizer):
    messages = (example["content"])
    try:
        #print("tokenized")
        tokenized_content = tokenizer.apply_chat_template(messages, tokenize=False)
        return {'content': tokenized_content}
    except Exception as e:
       #print(f"Unable to tokenize:")
        text =  "None"
        return {'content': " "}
    
import re
import random
from multiprocessing import cpu_count
column_names = ['scene','content']
raw_datasets = datasets_train_test.map(apply_chat_template,
                                num_proc=cpu_count(),
                                fn_kwargs={"tokenizer": tokenizer},
                                remove_columns=column_names,
                                desc="Applying chat template",)

train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]


from transformers import BitsAndBytesConfig
import torch

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
)


model_kwargs = dict(
    attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
    torch_dtype="auto",
    use_cache=False, # set to False as we're going to use gradient checkpointing
    device_map="auto",
    quantization_config=quantization_config,
)

from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments

# path where the Trainer will save its checkpoints and logs
output_dir = '/home/csgrad/sunilruf/emotion_chatbot/master/sunils_code/mistra/rank_64/rank_64_2/'

# based on config
training_args = TrainingArguments(
    fp16=True, # specify bf16=True instead when training on GPUs that support bf16
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=128,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-04,
    log_level="info",
    logging_steps=5,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=50,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_eval_batch_size=1, # originally set to 8
    per_device_train_batch_size=2, # originally set to 8
    # push_to_hub=True,
    # hub_model_id="zephyr-7b-sft-lora",
    # hub_strategy="every_save",
    report_to="tensorboard",
    save_strategy="epoch",
    logging_dir=output_dir+'logs/',
    save_total_limit=4,
    seed=42,
    
)
print("passed training_args")
# based on config
peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

def preprocess_dataset(dataset):
    for sample in dataset:
        # Convert 'content' field to string
        sample['content'] = str(sample['content'])
    return dataset

# Preprocess datasets
train_dataset = preprocess_dataset(train_dataset)
eval_dataset = preprocess_dataset(eval_dataset)

tokenizer.pad_token = tokenizer.eos_token

# Then, create the trainer
trainer = SFTTrainer(
        model=model,
        #model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="content",
        tokenizer=tokenizer,
        packing=True,  # Disable packing
        peft_config=peft_config,
        max_seq_length=4096,
    )
print("Trainer staarted")
trainer.train()

trainer.save_model('/home/csgrad/sunilruf/emotion_chatbot/master/sunils_code/mistra/rank_64/')
print("done")