import streamlit as st
from utils.document_processing import load_documents, chunk_documents
from utils.vector_db_utils import create_vector_db, load_vector_db, retrieve_relevant_documents
from utils.llm_utils import setup_llm, create_qa_chain
from utils.translation_utils import translate_text, summarize_text
import os
import torch


torch.classes.__path__ = [] 


# Configuration
DATA_PATH = "data\Raw"
VECTOR_DB_PATH = "./data/vector_db"
LLM_MODEL_PATH = "models\llama-2-7b-chat.ggmlv3.q4_0.bin" 
TRANSLATION_MODEL_PATH = "models\opus-mt-en-fr" 


st.title("Mysterious Document Analyzer")

if st.button("Process Documents"):
    with st.spinner("Processing documents..."):
        documents = load_documents(DATA_PATH)
        print("Done loading ...")
        chunks = chunk_documents(documents)
        print("Done chunking ...")
        vector_db = create_vector_db(chunks, VECTOR_DB_PATH)
        print("Done create_vector_db ...")
        st.session_state.vector_db = vector_db
        print("session_state ...")
        llm = setup_llm(LLM_MODEL_PATH)
        print("llm set ...")
        st.session_state.qa_chain = create_qa_chain(llm, vector_db)
        st.success("Documents processed!")

if "qa_chain" in st.session_state:
    query = st.text_input("Ask a question about the documents:")
    if query:
        with st.spinner("Answering..."):
            answer = st.session_state.qa_chain({"query": query})["result"]
            st.write("Answer:", answer)
            relevant_docs = retrieve_relevant_documents(st.session_state.vector_db, query)
            st.write("Relevant Documents:", relevant_docs)

    translate_query = st.text_input("Translate some text:")
    if translate_query:
        translated_text = translate_text(translate_query, "en", "fr", TRANSLATION_MODEL_PATH)
        st.write("Translated Text:", translated_text)

    summarize_query = st.text_area("Summarize some text:")
    if summarize_query:
        llm = setup_llm(LLM_MODEL_PATH)
        summary = summarize_text(llm, summarize_query)
        st.write("Summary:", summary)