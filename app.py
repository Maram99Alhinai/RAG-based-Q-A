import streamlit as st
import os
import torch
import time
from pathlib import Path

# Set page configuration and styling
st.set_page_config(
    page_title="Mysterious Document Analyzer",
    page_icon="📚",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 30px;
        text-align: center;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #2563EB;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .success-message {
        background-color: #DCFCE7;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #22C55E;
    }
    .info-message {
        background-color: #EFF6FF;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #3B82F6;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)

# Prevent CUDA initialization error
torch.classes.__path__ = []

# Initialize session state variables if they don't exist
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'data_path' not in st.session_state:
    st.session_state.data_path = "data/Raw"
if 'vector_db_path' not in st.session_state:
    st.session_state.vector_db_path = "./data/vector_db"

# Paths for models
LLM_MODEL_PATH = "models/llama-2-7b-chat.ggmlv3.q4_0.bin"
TRANSLATION_MODEL_PATH = "models/opus-mt-en-fr"

# Main title
st.markdown('<p class="main-title">📚 Mysterious Document Analyzer</p>', unsafe_allow_html=True)

# Simplified sidebar
with st.sidebar:
    st.markdown("### Document Setup")
    
    # Option to select data directory
    data_option = st.radio(
        "Select data source:",
        ["Use default directory", "Upload files", "Custom directory"]
    )
    
    if data_option == "Use default directory":
        st.session_state.data_path = "data/Raw"
        st.info("Using default: data/Raw")
        
    elif data_option == "Upload files":
        uploaded_files = st.file_uploader("Upload files:", accept_multiple_files=True, type=['pdf', 'docx', 'csv', 'xlsx'])
        
        if uploaded_files:
            # Create a temporary directory to store uploaded files if it doesn't exist
            temp_dir = Path("data/uploaded")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save uploaded files to the temp directory
            for uploaded_file in uploaded_files:
                with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.session_state.data_path = str(temp_dir)
            st.success(f"{len(uploaded_files)} files uploaded")
            
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
    
    st.markdown("---")
    
    # Processing buttons - simplified
    process_option = st.radio(
        "Processing options:",
        ["Process documents", "Load existing database"]
    )
    
    if st.button("Start"):
        if process_option == "Process documents":
            if st.session_state.data_path:
                with st.spinner("Loading necessary modules..."):
                    # Only import when needed
                    from utils.document_processing import load_documents, chunk_documents
                    from utils.vector_db_utils import create_vector_db
                    from utils.llm_utils import setup_llm, create_qa_chain
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Load documents
                    status_text.text("Loading documents...")
                    progress_bar.progress(10)
                    documents = load_documents(st.session_state.data_path)
                    
                    # Chunk documents
                    status_text.text("Chunking documents...")
                    progress_bar.progress(30)
                    chunks = chunk_documents(documents)
                    
                    # Create vector database
                    status_text.text("Building vector database...")
                    progress_bar.progress(50)
                    vector_db = create_vector_db(chunks, st.session_state.vector_db_path)
                    st.session_state.vector_db = vector_db
                    
                    # Setup language model
                    status_text.text("Setting up language model...")
                    progress_bar.progress(70)
                    llm = setup_llm(LLM_MODEL_PATH)
                    st.session_state.llm = llm
                    
                    # Create QA chain
                    status_text.text("Initializing QA system...")
                    progress_bar.progress(90)
                    st.session_state.qa_chain = create_qa_chain(llm, vector_db)
                    
                    # Complete
                    progress_bar.progress(100)
                    status_text.markdown('<div class="success-message">✅ Documents processed!</div>', unsafe_allow_html=True)
                    st.session_state.documents_processed = True
                    
                    # Add small delay for better user experience
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please select a data directory first.")
        else:  # Load existing
            with st.spinner("Loading database..."):
                # Only import when needed
                from utils.vector_db_utils import load_vector_db
                from utils.llm_utils import setup_llm, create_qa_chain
                
                try:
                    # Load vector database
                    vector_db = load_vector_db(st.session_state.vector_db_path)
                    if vector_db:
                        st.session_state.vector_db = vector_db
                        
                        # Setup language model
                        llm = setup_llm(LLM_MODEL_PATH)
                        st.session_state.llm = llm
                        
                        # Create QA chain
                        st.session_state.qa_chain = create_qa_chain(llm, vector_db)
                        
                        st.session_state.documents_processed = True
                        st.success("Database loaded!")
                        st.rerun()
                    else:
                        st.error("Database not found.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Main content area
if st.session_state.documents_processed:
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Question Answering", "Translation", "Summarization"])
    
    with tab1:
        st.markdown('<p class="section-header">Ask Questions About Your Documents</p>', unsafe_allow_html=True)
        
        query = st.text_input("Your question:", key="qa_query")
        show_sources = st.checkbox("Show source documents", value=True)
        
        if query:
            print(query)
            with st.spinner("Analyzing documents..."):
                # Ensure necessary modules are imported
                if "retrieve_relevant_documents" not in globals():
                    from utils.vector_db_utils import retrieve_relevant_documents
                
                # Get answer
                answer = st.session_state.qa_chain.invoke({"query": query})["result"]
                

                # In your app.py, within the "Question Answering" tab:
                st.markdown("### Answer")
                st.markdown(f"<div style='background-color:#E6EEF7; color: black; padding:15px; border-radius:5px;'>{answer}</div>", unsafe_allow_html=True)
                
                # Show relevant documents if checkbox is selected
                if show_sources and st.session_state.vector_db:
                    st.markdown("### Relevant Sources")
                    relevant_docs = retrieve_relevant_documents(st.session_state.vector_db, query)
                    
                    for i, doc in enumerate(relevant_docs):
                        with st.expander(f"Source {i+1}"):
                            st.markdown(doc.page_content)
                            if hasattr(doc.metadata, 'source') and doc.metadata.source:
                                st.caption(f"Source: {doc.metadata.source}")
    
    with tab2:
        st.markdown('<p class="section-header">Translate Documents</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            source_lang = st.selectbox("From", ["English", "French", "Spanish", "German"], index=0)
        with col2:
            target_lang = st.selectbox("To", ["French", "English", "Spanish", "German"], index=0)
        
        translate_text_input = st.text_area("Text to translate:", height=150, key="translate_input")
        
        if translate_text_input:
            with st.spinner("Translating..."):
                # Only import when needed
                if "translate_text" not in globals():
                    from utils.translation_utils import translate_text
                
                # Map language names to codes
                lang_codes = {
                    "English": "en",
                    "French": "fr",
                    "Spanish": "es",
                    "German": "de"
                }
                
                translated_text = translate_text(
                    translate_text_input, 
                    lang_codes[source_lang], 
                    lang_codes[target_lang], 
                    TRANSLATION_MODEL_PATH
                )
                
                st.markdown("### Translation")
                st.markdown(f"<div style='background-color:#F9FAFB; padding:15px; border-radius:5px;'>{translated_text}</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<p class="section-header">Summarize Text</p>', unsafe_allow_html=True)
        
        summarize_text_input = st.text_area("Text to summarize:", height=200, key="summarize_input")
        summary_length = st.slider("Summary length", min_value=1, max_value=5, value=3)
        
        if summarize_text_input:
            with st.spinner("Generating summary..."):
                # Only import when needed
                if "summarize_text" not in globals():
                    from utils.translation_utils import summarize_text
                
                summary = summarize_text(st.session_state.llm, summarize_text_input, length=summary_length)
                
                st.markdown("### Summary")
                st.markdown(f"<div style='background-color:#F9FAFB; padding:15px; border-radius:5px;'>{summary}</div>", unsafe_allow_html=True)
else:
    # Display welcome message and instructions when no documents are processed
    st.markdown("""
    ## Welcome to Mysterious Document Analyzer! 👋
    
    This application helps you analyze, search, translate, and summarize your documents.
    
    ### Getting Started
    
    1. Select your data source in the sidebar
    2. Click "Start" to process your documents
    3. Use the tabs above to interact with your documents
    
    ### Supported Document Types
    - PDF files (.pdf)
    - Word documents (.docx)
    - CSV files (.csv)
    - Excel spreadsheets (.xlsx)
    """)
    
    # Display some nice graphics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📄 Document Processing")
        st.markdown("Process multiple document formats")
    with col2:
        st.markdown("### 🔍 Semantic Search")
        st.markdown("Find information using natural language")
    with col3:
        st.markdown("### 🌐 Translation & Summary")
        st.markdown("Translate text and create summaries")

# Footer
st.markdown("---")
st.markdown("Mysterious Document Analyzer © 2025")