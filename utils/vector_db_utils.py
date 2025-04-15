from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import AtlasDB
import os

def create_vector_db(chunks, atlas_token, atlas_project_name):
    """Creates a vector database using Nomic Atlas."""
    embeddings = HuggingFaceEmbeddings()
    vector_db = AtlasDB.from_documents(
        chunks,
        embeddings,
        atlas_token=atlas_token,
        project_name=atlas_project_name,
        distance_metric="cosine",
    )
    return vector_db

def retrieve_relevant_documents(vector_db, query):
    """Retrieves relevant documents from the vector database."""
    relevant_docs = vector_db.similarity_search(query)
    return relevant_docs