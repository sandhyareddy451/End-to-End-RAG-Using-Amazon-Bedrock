from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import boto3
import streamlit as st

import os
from dotenv import load_dotenv


load_dotenv()


aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")
region_name = os.getenv("region_name")


# boto3 is used to make the connection with bedrock

#bedrcok client

bedrock_client = boto3.client(
    
    service_name ="bedrock-runtime",
    aws_access_key_id= aws_access_key_id,
    aws_secret_access_key = aws_secret_access_key,
    region_name=region_name
)

model_id = "mistral.mistral-large-2402-v1:0"

llm = Bedrock(
    
    model_id = model_id,
    client = bedrock_client,
    model_kwargs = {"temperature": 0.9}
    
)


def my_chatbot(language, user_text):
    prompt = PromptTemplate(
        
        input_variables = ['language', 'user_text'],
        template = "you are an chat bot, you are in {language}.\n\n {user_text}"
        
    )
    
    bedrock_chain = LLMChain(llm=llm, prompt= prompt)
    response = bedrock_chain.invoke({'language': language, 'user_text': user_text})
    return response

st.title("Test Application Using Amazon BEDROCK ")

language = st.sidebar.selectbox("language", ['Hindi', "English", "Spanish"])

if language:
    user_text = st.sidebar.text_area(label = "please type your Question?", max_chars = 100)

if user_text:
    response = my_chatbot(language, user_text)
    st.write(response['text'])