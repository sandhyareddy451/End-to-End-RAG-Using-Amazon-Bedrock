import boto3
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_transformers import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import  PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import streamlit as st


load_dotenv()
aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")
region_name = os.getenv("region_name")


prompt_template = """


Human: Use the following peces of of context to provide a concise answer to the question
at the end but use atleast summarize with 250 words with detailed explainations. 
if you don't know the answer, just say that you don't know, don't try to make up answer.

<context>
{context}
</context>

Question : {question}

Assistant :"""

# Bedrock Clients

bedrock_client = boto3.client(
    
    service_name = "bedrock-runtime",
    aws_access_key_id = aws_access_key_id,
    aws_secret_access_key = aws_secret_access_key,
    region_name = region_name
)


# Get Embed model

bedrock_embeddings = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1", client = bedrock_client)


def get_documents():
    loader = PyPDFDirectoryLoader("pdf_data")
    documnets = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 500)
    
    
    docs = text_splitter.split_documnets(documnets)
    return docs

def get_vector_store(docs):
        vectorstore_faiss = FAISS.from_documnets(
            
            docs,
            bedrock_embeddings
        )
        vectorstore_faiss.save_local("faiss_index")
        
        
def get_llm():
    llm = Bedrock(model_id = "mistral.mistral-large-2402-v1:0", client = bedrock_client,
        model_kwargs ={'max_gen_len': 512})
    
    return llm
    

prompt = PromptTemplate(
    
    
    template = prompt_template,
    input_variables = ["context",'question']
    
    
    
    )

def get_response_llm (llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        
    llm = llm,
    chain_type = "stuff",
    retriever = vectorstore_faiss.as_retriever(
        search_type = "similarity", search_kwargs = {"k":3}
        
    ),
    
)
    answer =qa({"query":query})
    return answer['result']

def main():
    
    st.set_page_config("RAG Demo")
    st.header("End to End RAG Application")
    user_question = st.text_iput("Ask a Question from the PDF Files")
    
    
    
    with st.sidebar:
        st.title("update or create vectore store")
        
        if st.button("store Vectore"):
            with st.spinner("Processing..."):
                docs = get_documents()
                get_vector_store(docs)
                st.sucess("done")

       if st.button("Send"):
            with st.spinner("Processing..."):
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_llm()
                st.write(get_response_llm(llm,faiss_index,user_question))


                st.write(get_response_llm(llm, faiss_index, user_question))
                
                
                
if __name__ == "__main__":
    main()
