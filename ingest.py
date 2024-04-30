import streamlit as st
import os
import nest_asyncio  # noqa: E402

nest_asyncio.apply()

# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv

load_dotenv()

##### LLAMAPARSE #####
from llama_parse import LlamaParse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# from fastembed.embedding import DefaultEmbedding
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import DirectoryLoader
import pickle

llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown(
    """
## Document Genie: Get instant insights from your Documents

This chatbot is processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

3. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
"""
)


def apload_data(uploaded_data, list_to_parse):
    file_paths = "./data/raw"
    if not os.path.exists(file_paths):
        os.makedirs(file_paths)
    for file in uploaded_data:
        file_name = os.path.join(file_paths, file.name)
        with open(file_name, "wb") as f:
            f.write(file.getbuffer())
        list_to_parse.append(file_name)
        st.success(f"List_to_parse: {list_to_parse}")
    return list_to_parse


def load_or_parse_data(list_to_parse):
    data_file = "./data/parsed_data.pkl"

    if os.path.exists(data_file):
        # Load the parsed data from the file
        with open(data_file, "rb") as f:
            parsed_data = pickle.load(f)
    else:
        lama_parse_documents = LlamaParse(
            api_key=llamaparse_api_key, result_type="markdown"
        ).load_data(list_to_parse)

        # Save the parsed data to a file
        with open(data_file, "wb") as f:
            pickle.dump(lama_parse_documents, f)

        # Set the parsed data to the variable
        parsed_data = lama_parse_documents

    return parsed_data


def create_vector_database(lama_parse_documents):
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    # Call the function to either load or parse the data

    with open("data/output.md", "a") as f:  # Open the file in append mode ('a')
        for doc in lama_parse_documents:
            f.write(doc.text + "\n")

    loader = DirectoryLoader("data/", glob="**/*.md", show_progress=True)
    documents = loader.load()
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Initialize Embeddings
    embeddings = FastEmbedEmbeddings()

    # Create and persist a Chroma vector database from the chunked documents
    qdrant = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        url=qdrant_url,
        collection_name="rag",
        api_key=qdrant_api_key,
    )

    print("Vector DB created successfully !")


def create_database():
    st.header("AI clone chatbot💁")

    user_question = st.text_input(
        "Ask a Question from the PDF Files", key="user_question"
    )

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        uploaded_data = st.file_uploader(
            "Please upload data in format .pdf", type="pdf", accept_multiple_files=True
        )
        if st.button("Submit & Process", key="process_button"):

            with st.spinner("Processing..."):
                list_to_parse = []
                if uploaded_data is not None:
                    list_to_parse = apload_data(uploaded_data, list_to_parse)
                    # Call the function to either load or parse the data
                    lama_parse_documents = load_or_parse_data(list_to_parse)
                    create_vector_database(lama_parse_documents)


if __name__ == "__main__":
    create_database()
