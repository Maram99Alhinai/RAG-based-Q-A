from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

def setup_llm(model_path):
    """Sets up the local LLM."""
    llm = CTransformers(model=model_path, model_type="llama")
    return llm

def create_qa_chain(llm, vector_db):
    """Creates a RAG-based Q&A chain."""
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())
    return qa_chain