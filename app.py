from utils.llm_utils import create_qa_chain ,process_documents_and_create_db ,setup_llm_and_qa
from utils.vector_db_utils import load_vector_db , retrieve_relevant_documents
from utils.document_processing import load_document
from utils.translation_utils import translate
from pathlib import Path
import streamlit as st
import torch
import os



# --- CONFIGURATION ---
st.set_page_config(page_title="Mysterious Document Analyzer", page_icon="📚", layout="wide")
torch.classes.__path__ = []  # Prevent CUDA error (source: [PyTorch Forums](https://discuss.pytorch.org/))
LLM_MODEL_PATH = "models/llama-2-7b-chat.ggmlv3.q4_0.bin"
TRANSLATION_MODEL_PATH = "models/m2m100_418M"

# --- SESSION STATE ---
for key, value in {
    "vector_db": None,
    "llm": None,
    "qa_chain": None,
    "documents_processed": False,
    "data_path": "data/Raw",
    "vector_db_path": "./data/vector_db",
    "translation_service": None
}.items():
    st.session_state.setdefault(key, value)

# --- STYLE ---
if Path("style.css").exists():
    st.markdown(f"<style>{Path('style.css').read_text()}</style>", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<p class="main-title">📚 Mysterious Document Analyzer</p>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Document Analysis", "Translation"])


# --- TAB 1: Document Analysis ---
with tab1:
    with st.sidebar:
        st.markdown("### Setup")
        load_option = st.radio("Choose setup option:", ["Process New Documents", "Load Existing Database"])

        if load_option == "Process New Documents":
            data_option = st.radio("Select data source:", ["Use default directory", "Upload files", "Custom directory"])

            if data_option == "Use default directory":
                st.session_state.data_path = "data/Raw"
                st.info("Using default: data/Raw")

            elif data_option == "Upload files":
                uploaded_files = st.file_uploader("Upload files:", type=['pdf', 'docx', 'csv', 'xlsx'], accept_multiple_files=True)
                if uploaded_files:
                    temp_dir = Path("data/uploaded")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    for file in uploaded_files:
                        Path(temp_dir, file.name).write_bytes(file.getbuffer())
                    st.session_state.data_path = str(temp_dir)
                    st.success(f"{len(uploaded_files)} files uploaded.")

            elif data_option == "Custom directory":
                st.session_state.data_path = st.text_input("Enter directory path:", value=st.session_state.data_path)
                if st.button("Browse"):
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk()
                    root.withdraw()
                    selected_dir = filedialog.askdirectory()
                    if selected_dir:
                        st.session_state.data_path = selected_dir
                        st.rerun()

            if st.button("Preprocess Documents"):
                with st.spinner("Processing documents..."):
                    vector_db = process_documents_and_create_db(st.session_state.data_path, st.session_state.vector_db_path)
                    if vector_db:
                        st.session_state.vector_db = vector_db
                        st.session_state.llm, st.session_state.qa_chain = setup_llm_and_qa(vector_db, LLM_MODEL_PATH)
                        st.session_state.documents_processed = True
                        st.rerun()

        else:  # Load Existing Database
            if st.button("Load Database"):
                with st.spinner("Loading database..."):
                    vector_db = load_vector_db(st.session_state.vector_db_path)
                    if vector_db:
                        st.session_state.vector_db = vector_db
                        st.session_state.llm, st.session_state.qa_chain = setup_llm_and_qa(vector_db, LLM_MODEL_PATH)
                        st.session_state.documents_processed = True
                        st.success("Database loaded!")
                        st.rerun()
                    else:
                        st.error("Database not found. Please process documents first.")

        # Allow adding files to existing database
        if st.session_state.vector_db:
            st.markdown("---")
            uploaded_files_add = st.file_uploader("Upload files to add:", type=['pdf', 'docx', 'csv', 'xlsx'], accept_multiple_files=True)
            if uploaded_files_add and st.button("Add Files"):
                with st.spinner("Adding files to database..."):
                    temp_dir = Path("data/uploaded_add")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    for file in uploaded_files_add:
                        Path(temp_dir, file.name).write_bytes(file.getbuffer())

                    if process_documents_and_create_db(str(temp_dir), st.session_state.vector_db_path):
                        st.session_state.vector_db = load_vector_db(st.session_state.vector_db_path)
                        st.session_state.qa_chain = create_qa_chain(st.session_state.llm, st.session_state.vector_db)
                        st.success(f"{len(uploaded_files_add)} files added.")
                        st.rerun()
                    else:
                        st.error("Failed to add new files.")

    # --- MAIN CONTENT (Question Answering) ---
    if st.session_state.documents_processed and st.session_state.qa_chain:
        st.markdown('<p class="section-header">Ask Questions About Your Documents</p>', unsafe_allow_html=True)

        query = st.text_input("Your question:", key="qa_query")
        show_sources = st.checkbox("Show source documents", value=True)

        if query:
            with st.spinner("Analyzing documents..."):
                answer = st.session_state.qa_chain.invoke({"query": query})["result"]
                st.markdown("### Answer")
                st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

                if show_sources:
                    st.markdown("### Relevant Sources")
                    for i, doc in enumerate(retrieve_relevant_documents(st.session_state.vector_db, query)):
                        with st.expander(f"Source {i+1}"):
                            st.markdown(doc.page_content)
                            if hasattr(doc.metadata, 'source'):
                                st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")

    else:
        st.markdown("""
        ## Welcome to Mysterious Document Analyzer! 👋
        1. Choose to process new documents or load an existing database.
        2. Preprocess or load your documents.
        3. Ask questions based on your document content!
        """)

# --- TAB 2: Translation ---
with tab2:
    st.markdown('<p class="section-header">Upload and Translate Documents</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload file(s) to translate:", type=['pdf', 'docx', 'txt', 'csv', 'xlsx'], accept_multiple_files=True, key="upload_translation")
    source_lang = st.selectbox("Source language:", ["en", "de", "fr", "es", "ar", "zh"])
    target_lang = st.selectbox("Target language:", ["en", "de", "fr", "es", "ar", "zh"])

    if uploaded_files:
        temp_dir = Path("data/uploaded")
        temp_dir.mkdir(parents=True, exist_ok=True)
        for file in uploaded_files:
            file_path = Path(temp_dir, file.name)
            file_path.write_bytes(file.getbuffer())

            if st.button("Translate Documents"):
                with st.spinner("Translating documents..."):
                    documents = load_document(str(file_path))
                    if documents:
                        for doc in documents:
                            text = doc.page_content[:200]  # Only translate snippet
                            translation = translate(text, model_path=TRANSLATION_MODEL_PATH, source_lang=source_lang, target_lang=target_lang)
                            st.markdown(f"### 📄 {doc.metadata.get('source', 'Unknown File')}")
                            st.markdown(f"<div class='translation-box'>{translation}</div>", unsafe_allow_html=True)
                    else:
                        st.error("No documents loaded.")
    else:
        st.info("Please upload documents to translate.")

# --- FOOTER ---
st.markdown("---")
st.caption("Mysterious Document Analyzer © 2025")