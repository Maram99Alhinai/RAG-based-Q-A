from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vector_db(chunks, persist_directory):
    """Creates a vector database using FAISS."""
    # Create embeddings model
    embeddings = HuggingFaceEmbeddings()
    
    # Create FAISS vector store from documents
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # Save the vector store to disk
    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
        vector_db.save_local(persist_directory)
    
    return vector_db

def load_vector_db(persist_directory):
    """Loads an existing vector database from disk."""
    embeddings = HuggingFaceEmbeddings()
    
    if os.path.exists(persist_directory):
        vector_db = FAISS.load_local(persist_directory, embeddings,
                                     allow_dangerous_deserialization=True)
        return vector_db
    else:
        return None

def retrieve_relevant_documents(vector_db, query, k=1):
    """Retrieves relevant documents from the vector database."""
    relevant_docs = vector_db.similarity_search(query, k=k)
    return relevant_docs