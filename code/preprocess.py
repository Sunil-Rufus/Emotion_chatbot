import torch
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ['TRANSFORMERS_CACHE'] = '/home/csgrad/sunilruf/'
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.cuda.empty_cache()
device = "cuda" # the device to load the model onto

json_file_path = '../../../ExTES.json'  # Change 'output.json' to your desired file name

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

df.to_csv('../data/processed_data.csv', index=False)

