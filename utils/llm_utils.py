from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import streamlit as st



def setup_llm(model_path):
    """Sets up the local LLM."""
    llm = CTransformers(model=model_path, model_type="llama")
    return llm


def create_qa_chain(llm, vector_db):
    """Creates a RAG-based Q&A chain."""
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=vector_db.as_retriever())
    return qa_chain


def setup_llm_and_qa(vector_db, LLM_MODEL_PATH):
    """Sets up the LLM and creates the QA chain."""
    llm = setup_llm(LLM_MODEL_PATH)
    qa_chain = create_qa_chain(llm, vector_db)
    return llm, qa_chain
    

def process_documents_and_create_db(data_path, vector_db_path):
    """Loads, chunks, and creates a vector database."""
    from utils.document_processing import load_documents, chunk_documents
    from utils.vector_db_utils import create_vector_db

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Load documents
        status_text.text("Loading documents...")
        progress_bar.progress(10)
        documents = load_documents(data_path)

        # Chunk documents
        status_text.text("Chunking documents...")
        progress_bar.progress(50)
        chunks = chunk_documents(documents)

        # Create vector database
        status_text.text("Building vector database...")
        progress_bar.progress(80)
        vector_db = create_vector_db(chunks, vector_db_path)
        progress_bar.progress(100)
        status_text.markdown('<div class="success-message">✅ Documents processed and vector database created!</div>', unsafe_allow_html=True)
        return vector_db

    except Exception as e:
        st.error(f"Error during document processing: {str(e)}")
        return None
