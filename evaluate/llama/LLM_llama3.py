import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings # type: ignore
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from typing import Union
from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
import time
import subprocess
import qrcode
from pdf2image import convert_from_bytes
from langchain.llms import HuggingFacePipeline
from torch import cuda, bfloat16
import transformers
#import timm

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ["TRANSFORMERS_CACHE"] = '/data/'
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"
images1 = convert_from_bytes(open(
    'LLM_code/data/grad-handbook-2023.pdf', 'rb').read())

documents = []
scraped_files_path = "LLM_code/docs/"
handbook_path = "LLM_code/data/"
for file in os.listdir(scraped_files_path):
    if file.endswith('.txt'):
        text_path = scraped_files_path+ file
        loader = TextLoader(text_path)
        documents.extend(loader.load())
for file in os.listdir(handbook_path):
    if file.endswith(".pdf"):
        pdf_path = handbook_path + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
        

#split the data into chunks
text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=20)
documents = text_splitter.split_documents(documents)

#load the retriever model to create embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")#"hkunlp/instructor-base")
vectordb = FAISS.from_documents(documents, embeddings)

#context book updation if needed
try:
    print("Entered context handbook updation")
    with open('../context_handbook.txt', 'r') as file:
        data = file.read().replace('\n', '')
except:
    print("No data in context_handbook")
try:
    vectordb.add_texts([str(data)])
except:
    print("Context handbook added to vectordb")


#load the model for generation
model_id = 'meta-llama/Llama-2-13b-chat-hf'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config, 
    #quantization_config=bnb_config,
    #device="cuda:0",
    device_map='auto',
    use_auth_token=hf_auth
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)



#inititalise the pipeline
generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    #stopping_criteria=stopping_criteria,  # without this model rambles during chat
    do_sample=True,
    temperature=0.2,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=4096,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

llm = HuggingFacePipeline(pipeline=generate_text)


#retrieve the results
pdf_qa = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_type = "similarity_score_threshold", search_kwargs={'score_threshold': 0.1, 'k': 4}),
    max_tokens_limit=1024,
    return_source_documents=True,
    verbose=False
)


from respond import LLMResponse
#query = " "
LLMResponse(query, pdf_qa, images1)

