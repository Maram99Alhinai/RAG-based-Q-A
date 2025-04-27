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


def load_document(data_path):
    """Loads a single document from the specified path."""
    document = []
    file_name = os.path.basename(data_path) # gets the name of the file.

    if data_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(data_path)
    elif data_path.endswith(".pdf"):
        loader = PyPDFLoader(data_path)
    elif data_path.endswith(".xlsx"):
        loader = UnstructuredExcelLoader(data_path)
    else:
        raise ValueError(f"File type not supported: {data_path}")  # Raise an error for unsupported file types

    document = loader.load()  # Load the document
    for doc in document:
        doc.metadata['source'] = file_name # Preserve the filename in the metadata.
    return document



def chunk_documents(documents, chunk_size=200, chunk_overlap=10):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
