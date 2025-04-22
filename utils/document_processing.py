import os
from langchain_community.document_loaders import (UnstructuredWordDocumentLoader, 
                                                  PyPDFLoader ,UnstructuredExcelLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter



def load_documents(data_path):
    """Loads documents from the specified path."""
    documents = []
    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)
        if filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath)
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(filepath)
        else:
            error="This file type is not supported"
            print(error)
            continue
        documents.extend(loader.load())
    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=20):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
