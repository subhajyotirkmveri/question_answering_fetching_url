
# QuestionAnsweringBot: Ask any question and fetch the answer from relevant URLs

![](rockybot.jpg)

## Features
- Ask any question and returns that relevant urls
- From those urls select some url as your wish
- Load URLs 
- Process article content through LangChain's UnstructuredURL Loader
- Construct an embedding vector using OpenAI's embeddings and leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of relevant information
- Interact with the LLM's  by inputting queries and receiving answers along with source URLs.


## 
# Configuring
First create a virtual environemnt.
```
python -m venv env
```
or 
```
conda create -n "your_environment_name"
```
secondly activate the virtual environment
```
conda activate "your_environment_name"
```
1.Clone this repository to your local machine using:

```
git clone https://github.com/subhajyotirkmveri/question_answering_fetching_url.git
```
2.Navigate to the project directory:

```bash
  cd question_answering_fetching_url
```
3. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
4. Need to dwonload the llama2 model or mistral model from the below link
```
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin
```
or 
```
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main
```
or 

```
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main
```
5. create a folder named 'models' at the same level as the Python files and  put the download model into 'models' folder
## Usage/Examples

1. Run the Streamlit app by executing:
```bash
streamlit run main_1.py

```

2.The web app will open in your browser.

- Configuration File Generator:
  - RETURN_SOURCE_DOCUMENTS: Select whether to return source documents (True/False)
  - VECTOR_COUNT: Select the number of vectors (1, 2, 3).
  - CHUNK_SIZE: Select the chunk size for text splitting (range: 50 to 1000).
  - CHUNK_OVERLAP: Select the chunk overlap for text splitting (range: 0 to 50)
  - DB_FAISS_PATH: Set the FAISS database path
  - MODEL_TYPE: Choose the model type (e.g., llama, mistral).[always remember if you choose llama type then you have to choose llama model]
  - MODEL_BIN_PATH: Set the model binary path
  - EMBEDDINGS: Select the embeddings model.
  - MAX_NEW_TOKENS: Select the maximum number of new tokens (512, 1024, 2048)
  - TEMPERATURE: Select the temperature for the model (0.00 to 1.00)

- Save Configuration:
  - Save the configuration to a YAML file
- Input Search Query
   - Enter a search query and specify the number of URLs to retrieve
- Search URLs:
  - Perform a Google search for the given query and display the retrieved URLs.
- Select URLs
  - Select specific URLs from the search results for further processing
- Process URLs
  -  Observe the system as it performs text splitting, generates embedding vectors, and efficiently indexes them using FAISS.
  - Display the progress at each step.
- Retrieve and Display Answers




## Project Structure

- main_1.py: The main Streamlit application script.
- requirements.txt: A list of required Python packages for the project.

