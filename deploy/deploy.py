import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from huggingface_hub import login


"""os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TRANSFORMERS_CACHE'] = '/home/csgrad/sunilruf/'"""
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<your_hf_access_token>"

login(token="<your_hf_access_token>")
@st.cache_resource
def load_model_and_tokenizer():
    
    path = "sunilrufus/Emotion_support_chatbot"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, load_in_4bit=True, device_map="auto")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
st.title("Emo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    input_ids = tokenizer.apply_chat_template(st.session_state.messages, truncation=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    tokenizer.pad_token = tokenizer.eos_token
    # inference
    outputs = model.generate(
            input_ids=input_ids,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
    )
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    value = (tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("[/INST]")[-1])
    value = value.encode('ascii', 'ignore').decode('ascii')
    #response = f"Echo: {value}"
    #st.session_state.messages.append({'role': 'assistant', 'content': value})
    print(st.session_state.messages)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(value)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": value})