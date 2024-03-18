import streamlit as st
import os
import json
import pandas as pd
from urllib.parse import unquote
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set environment variables for API keys
# Assuming 'userdata' is a dictionary-like object that holds your API keys
os.environ["HF_KEY"] = 'your_openai_api_backup_key'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = 'your_huggingface_hub_api_token'

# Streamlit UI
st.title('Document Processing and Vector Store Creation')

# Display the current working directory
cwd = os.getcwd()
st.write("Current working directory:", cwd)

# Set the directory you want to list files from
source_dir = './data'

st.title('Review Files in Directory')

# List files in the specified directory
if os.path.exists(source_dir):
    files = os.listdir(source_dir)
    if files:
        st.write('Files in directory:', source_dir)
        for file in files:
            st.text(file)
    else:
        st.write('No files found in the directory.')
else:
    st.error(f'The specified directory {source_dir} does not exist.')

# Construct the path to the paper_sources.csv file within that directory
csv_file_path = os.path.join(source_dir, 'paper_sources.csv')


st.title('Document Processing and Vector Store Creation')

# Check if the CSV file exists and then load it
if os.path.exists(csv_file_path):
    description_df = pd.read_csv(csv_file_path)
    
    # Decode URLs to regular filenames and create a new column 'Filename'
    description_df['Filename'] = description_df['URL'].apply(lambda x: unquote(os.path.basename(x)))
    
    st.write("Description CSV file loaded successfully.")
    st.dataframe(description_df)

    if st.button('Process Data'):
        EMBEDDING_MODEL_NAME = "thenlper/gte-small"
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        processed_texts = []
        for index, row in description_df.iterrows():
            pdf_path = os.path.join(source_dir, row['Filename'])
            if os.path.exists(pdf_path):
                # Placeholder for document processing logic
                # Replace the next line with actual text extraction from the PDF
                doc_text = "Sample extracted text here"
                processed_texts.append(doc_text)
                st.write(f"Processed {row['Filename']}")
            else:
                st.error(f"File not found: {pdf_path}")

        if processed_texts:
            vectorstore = FAISS.from_texts(processed_texts, embedding=embedding_model)
            # Persist the vector store
            processed_data_name = st.text_input('Name for Processed Data File', value='processed_data.json')
            vectorstore_name = st.text_input('Name for Vectorstore File', value='vectorstore.faiss')

            if st.button('Save Processed Data and Vectorstore'):
                with open(processed_data_name, 'w') as f:
                    json.dump(processed_texts, f)
                
                vectorstore.persist(vectorstore_name)
                st.success(f'Processed Data and Vectorstore saved as {processed_data_name} and {vectorstore_name}, respectively.')
else:
    st.error(f"Description file not found at {csv_file_path}")