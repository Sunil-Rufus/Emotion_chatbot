# Empowering Emotional Support Chatbot using LLM
The emergence of AI-driven chatbots presents a promising avenue for extending empathy
and support to individuals navigating emotional distress. This study proposes the utilization
of deep learning and natural language processing (NLP) methodologies to develop an
AI-driven emotional support chatbot. Specifically tailored to cultivate a supportive environment
for users encountering challenging emotional experiences, this chatbot aims to leverage
Large Language Models with significantly fewer parameters than contemporary state-of-theart
models, such as ChatGPT. Through the application of fine-tuning techniques on newer
datasets, this research endeavors to explore the inherent capabilities of language models in
delivering nuanced emotional support across diverse scenarios. Central to its objectives is the
refinement of emotional support chatbots by means of fine-tuning existing language models
on datasets curated for emotional comprehension. Furthermore, this study undertakes an
investigation into knowledge pruning techniques, with the goal of reducing the size and complexity
of trainable parameters within these models while ensuring the preservation of their
performance metrics. To ascertain the efficacy and reliability of the proposed methodologies,
evaluation procedures are conducted on standard datasets. By systematically testing
the efficiency of these enhanced emotional support chatbots, this research contributes to the
advancement of AI-driven solutions in the realm of mental health support services, positioning
them as integral components within the evolving landscape of digital care provision.

## Installation

1. Clone the repository
   ```
   git clone https://github.com/Sunil-Rufus/Emotion_chatbot.git
   cd Emotion_chatbot
   ```
2. Install requirements
   ```
   pip install requirements.txt
   ```

3. Run the chat application using Streamlit - add your huggingface API token to the code.
   ```
   cd deploy/
   streamlit run deploy.py
   ```
   
## Model and Dataset used for finetuning

The dataset used for finetuning can be found in [HuggingFace datasets](https://huggingface.co/datasets/sunilrufus/Extes_filtered1) 
The finetuned model can be found in [HuggingFace Hub](https://huggingface.co/sunilrufus/Emotion_support_chatbot)

     
