import streamlit as st
import os
import timeit
from langchain.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader

import box
import yaml


def load_config(config_file_path):
    with open(config_file_path, 'r', encoding='utf8') as ymlfile:
        cfg = box.Box(yaml.safe_load(ymlfile))
    return cfg

def create_vector_db(loader):
    cfg = load_config('config.yml')
    
    if not os.path.exists(cfg.DB_FAISS_PATH):
        os.makedirs(cfg.DB_FAISS_PATH)
        print("DB Fiass Folder created successfully")
    else:
        print("DB Faiss Folder already exists")
        
    documents = loader.load()
   
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size= cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)    
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS,
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(cfg.DB_FAISS_PATH)    

    
def load_llm():
    cfg = load_config('config.yml')
    llm = CTransformers(model=cfg.MODEL_BIN_PATH,
                        model_type=cfg.MODEL_TYPE
    )
    return llm    

# Define the prompt template
prompt_template = """Use the following pieces of information to answer the user's question.
    Try to provide as much text as possible from "response". If you don't know the answer, please just say 
    "I don't know the answer". Don't try to make up an answer.
    
    Context: {context},
    Question: {question}
    
    Only return correct and helpful answer below and nothing else.
    Helpful answer:
"""

def set_qa_prompt():
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt


def retrieval_qa_chain(llm, prompt,db):
    cfg = load_config('config.yml')
    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS,
                                       model_kwargs={'device': 'cpu'})

    #retriever =db.as_retriever(score_threshold=0.7)
    retriever=db.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT})
    
                                        
    chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=retriever,
                                           input_key="query",
                                           return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                           chain_type_kwargs={'prompt': prompt})
    return chain
    
def qa_chat():
    cfg = load_config('config.yml')
    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS,
                                       model_kwargs={'device': 'cpu'})
    db=FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_qa_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa   
    
def final_result(query):
    cfg = load_config('config.yml')
    qa_result = qa_chat()
    response = qa_result({"query":query})
    return response

