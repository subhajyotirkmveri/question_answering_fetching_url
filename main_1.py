import streamlit as st
from googlesearch import search
import os
import timeit
import shutil
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from base import load_config, final_result
import yaml

# Function to get URLs from Google search
def get_urls_from_search(query, num_urls):
    urls = []
    for url in search(query, num_results=num_urls):
        urls.append(url)
        if len(urls) == num_urls:
            break
    return urls

# Streamlit app
st.set_page_config(page_title='Question answeing ChatBot from fetch url', layout="wide", page_icon="📈", initial_sidebar_state="expanded")
st.title("Question answeing ChatBot from fetch url ")

# Ensure messages are initialized
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history at the top
st.header("Chat History")
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**User:** {message['content']}")
    elif message["role"] == "assistant":
        st.markdown(f"**Assistant:** {message['content']}")

# Settings sidebar
with st.sidebar:
    st.title("Settings")
    st.markdown('---')
    
    # Configuration File Generator
    st.title("Configuration File Generator")

    options = {
        "RETURN_SOURCE_DOCUMENTS": [True, False],
        "VECTOR_COUNT": [1, 2, 3],
        "CHUNK_SIZE": list(range(50, 1001)),
        "CHUNK_OVERLAP": list(range(0, 51)),
        "DB_FAISS_PATH": "db_faiss/",
        "MODEL_TYPE": ["llama", "mistral"],
        "MODEL_BIN_PATH": ['models/llama-2-7b-chat.ggmlv3.q8_0.bin', "models/Mistral-7B-Instruct-v0.1-GGUF/tree/main", "models/Mistral-7B-Instruct-v0.2-GGUF/tree/main"],
        "EMBEDDINGS": [ "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
        "MAX_NEW_TOKENS": [512, 1024, 2048],
        "TEMPERATURE": [round(i * 0.01, 2) for i in range(0, 101)]
    }

    config = {}

    for key, value in options.items():
        if isinstance(value, list):
            config[key] = st.selectbox(f"{key}:", value)
        else:
            config[key] = st.text_input(f"{key}:", value)

    if st.button("Save Configuration"):
        with open("config.yml", "w") as f:
            yaml.dump(config, f)
        st.success("Configuration saved successfully!")

# Input for the search query
search_query = st.text_input("Enter your query")

# Slider to select number of URLs
num_urls = st.slider("Number of URLs to retrieve", 1, 10, 1)

if st.button("Search URLs"):
    if search_query:
        urls = get_urls_from_search(search_query, num_urls)
        if urls:
            st.write(f"First {num_urls} URLs from Google search:")
            for i, url in enumerate(urls, start=1):
                st.write(f"{i}. {url}")
            st.session_state.urls = urls
        else:
            st.error("No results found for the query.")
    else:
        st.error("Please enter a query.")

selected_urls = []

if "urls" in st.session_state:
    for i, url in enumerate(st.session_state.urls, start=1):
        if st.checkbox(f"Select URL {i}", key=f"url_{i}"):
            selected_urls.append(url)

main_placeholder = st.empty()

if st.button("Find Answer") and selected_urls:
    cfg = load_config('config.yml')
    loader = UnstructuredURLLoader(urls= selected_urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    all_documents = loader.load()

    #all_documents = []
    
    #for url in selected_urls:
        #loader = UnstructuredURLLoader(urls=[url])
        #main_placeholder.text(f"Loading data from {url}...✅✅✅")
        #documents = loader.load()
        #all_documents.extend(documents)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)
        
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(all_documents)
    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS,
                                       model_kwargs={'device': 'cpu'})
                                       
    db = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    
    if not os.path.exists(cfg["DB_FAISS_PATH"]):
        os.makedirs(cfg["DB_FAISS_PATH"])
        print("DB Faiss Folder created successfully")
    else:
        print("DB Faiss Folder already exists")
        
    db.save_local(cfg["DB_FAISS_PATH"])    
    main_placeholder.text("Embedding Vector saved locally into desired folder...✅✅✅")

    start_time = timeit.default_timer()
    response = final_result(search_query)
    st.header(f"Answer from selected URLs")
    output = response['result']
    end_time = timeit.default_timer()
    st.write(output)
    st.write('=' * 50)
    st.write("Time to retrieve answer:", end_time - start_time, "sec")
    st.session_state.messages.append({"role": "user", "content": search_query})
    st.session_state.messages.append({"role": "assistant", "content": response["result"]})

if st.sidebar.button("Delete Faiss folder"):
    cfg = load_config('config.yml')
    
    if os.path.exists(cfg["DB_FAISS_PATH"]):
        shutil.rmtree(cfg["DB_FAISS_PATH"])
        st.sidebar.success("Faiss folder deleted successfully")
    else:
        st.sidebar.warning("Faiss folder does not exist")

