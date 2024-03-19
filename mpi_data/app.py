import streamlit as st
import os
import json
import pandas as pd
from urllib.parse import unquote


# Import necessary components for embeddings and PDF processing
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader


# Set API keys as environment variables
os.environ["HF_KEY"] = 'your_openai_api_backup_key'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = 'your_huggingface_hub_api_token'

def process_pdf_files(source_dir):
    processed_texts = []  # To hold processed texts from PDFs
    
    # List PDF files in the specified directory
    files = [f for f in os.listdir(source_dir) if f.endswith('.pdf')]
    
    for file in files:
        pdf_path = os.path.join(source_dir, file)
        try:
            # Load and process the PDF file
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            doc_text = " ".join(str(page) for page in pages)  # Convert each page to string if not already

            processed_texts.append(doc_text)
            print(f"Processed {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    return processed_texts


# Streamlit UI setup
st.title('Document Processing and Vector Store Creation')

# Display the current working directory
cwd = os.getcwd()
st.write("Current working directory:", cwd)

# Define source and output directories
source_dir = './data'
out_dir = os.path.join(source_dir, 'out')

# Ensure the output directory exists
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# List files in the specified directory
files = os.listdir(source_dir) if os.path.exists(source_dir) else []
if files:
    st.write('Files in directory:', source_dir)
    for file in files:
        st.text(file)
else:
    st.error(f'The specified directory {source_dir} does not exist or is empty.')

# Check for and process the description CSV file
csv_file_path = os.path.join(source_dir, 'paper_sources.csv')
if os.path.exists(csv_file_path):
    description_df = pd.read_csv(csv_file_path)
    description_df['Filename'] = description_df['URL'].apply(lambda x: unquote(os.path.basename(x)))
    st.write("Description CSV file loaded successfully.")
    st.dataframe(description_df)

    # Initialize embedding model
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
    )

    if st.button('Process Data'):
         # Process the PDF files in the source directory
        processed_texts = process_pdf_files(source_dir)

        # Create a vector store from the processed data
        vectorstore = FAISS.from_documents(processed_texts, embedding=embedding_model)

        retriever = vectorstore.as_retriever()

        # Here, you can continue with any other processing or use of embeddings
        # For demonstration, this code assumes `embeddings` are directly saved or used to create a vector store

        # Define filenames for saving
        processed_data_name = 'processed_data.json'
        vectorstore_name = 'vectorstore.faiss'

        # Save processed data and embeddings
        processed_data_path = os.path.join(out_dir, processed_data_name)
        vectorstore_path = os.path.join(out_dir, vectorstore_name)
        
        with open(processed_data_path, 'w') as f:
            json.dump(processed_texts, f)
        
        vectorstore.persist(vectorstore_path)

        st.success(f'Processed Data and Vectorstore saved as {processed_data_name} and {vectorstore_name} in the "out" folder.')
else:
    st.error(f"Description file not found at {csv_file_path}")