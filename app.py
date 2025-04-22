from utils.llm_utils import create_qa_chain ,process_documents_and_create_db ,setup_llm_and_qa
from utils.vector_db_utils import load_vector_db
from pathlib import Path
import streamlit as st
import torch
import os



# Set page configuration and styling
st.set_page_config(
    page_title="Mysterious Document Analyzer",
    page_icon="📚",
    layout="wide",
)


# Custom CSS for better styling
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


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


# Sidebar for setup
with st.sidebar:
    st.markdown("### Setup")

    load_option = st.radio(
        "Choose setup option:",
        ["Process New Documents", "Load Existing Database"]
    )

    if load_option == "Process New Documents":
        st.markdown("#### Process Documents")
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
                temp_dir = Path("data/uploaded")
                temp_dir.mkdir(parents=True, exist_ok=True)
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

        if st.button("Preprocess Documents"):
            with st.spinner("Processing documents..."):
                vector_db = process_documents_and_create_db(st.session_state.data_path, st.session_state.vector_db_path)
                if vector_db:
                    st.session_state.vector_db = vector_db
                    llm, qa_chain = setup_llm_and_qa(vector_db, LLM_MODEL_PATH)
                    st.session_state.llm = llm
                    st.session_state.qa_chain = qa_chain
                    st.session_state.documents_processed = True
                    st.rerun()

    elif load_option == "Load Existing Database":
        print(load_option)
        if st.button("Load Database"):
            print("load_option")
            with st.spinner("Loading database..."):
                vector_db = load_vector_db(st.session_state.vector_db_path)
                if vector_db:
                    st.session_state.vector_db = vector_db
                    print('session_state')
                    llm, qa_chain = setup_llm_and_qa(vector_db, LLM_MODEL_PATH)
                    print('setup_llm_and_qa')
                    st.session_state.llm = llm
                    st.session_state.qa_chain = qa_chain
                    print('qa_chain')
                    st.session_state.documents_processed = True
                    st.success("Database loaded!")
                    st.rerun()
                else:
                    st.error("Database not found. Please process documents first.")

    if st.session_state.vector_db:
        st.markdown("---")
        st.markdown("#### Add Files for Preprocessing")
        uploaded_files_add = st.file_uploader("Upload files to add to database:", accept_multiple_files=True, type=['pdf', 'docx', 'csv', 'xlsx'])
        if uploaded_files_add and st.button("Add Files"):
            with st.spinner("Adding files to database..."):
                temp_dir_add = Path("data/uploaded_add")
                temp_dir_add.mkdir(parents=True, exist_ok=True)
                new_files_path = str(temp_dir_add)
                for uploaded_file in uploaded_files_add:
                    with open(os.path.join(new_files_path, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                new_vector_db = process_documents_and_create_db(new_files_path, st.session_state.vector_db_path)
                if new_vector_db and st.session_state.llm:
                    st.session_state.vector_db = load_vector_db(st.session_state.vector_db_path) # Reload to include new data
                    st.session_state.qa_chain = create_qa_chain(st.session_state.llm, st.session_state.vector_db)
                    st.success(f"{len(uploaded_files_add)} files added to the database.")
                    st.rerun()
                elif new_vector_db and not st.session_state.llm:
                    st.error("LLM not initialized. Please preprocess initial documents first.")
                elif not new_vector_db:
                    st.error("Failed to add new files to the database.")

    st.markdown("---")

# Main content area for answering questions
if st.session_state.documents_processed and st.session_state.qa_chain:
    st.markdown('<p class="section-header">Ask Questions About Your Documents</p>', unsafe_allow_html=True)

    query = st.text_input("Your question:", key="qa_query")
    show_sources = st.checkbox("Show source documents", value=True)

    if query:
        print(query)
        with st.spinner("Analyzing documents..."):
            from utils.vector_db_utils import retrieve_relevant_documents
            print('working on ans')
            answer = st.session_state.qa_chain.invoke({"query": query})["result"]

            st.markdown("### Answer")
            st.markdown(f"<div style='background-color:#E6EEF7; color: black; padding:15px; border-radius:5px;'>{answer}</div>", unsafe_allow_html=True)

            if show_sources and st.session_state.vector_db:
                st.markdown("### Relevant Sources")
                relevant_docs = retrieve_relevant_documents(st.session_state.vector_db, query)

                for i, doc in enumerate(relevant_docs):
                    with st.expander(f"Source {i+1}"):
                        st.markdown(doc.page_content)
                        if hasattr(doc.metadata, 'source') and doc.metadata.source:
                            st.caption(f"Source: {doc.metadata.source}")

elif not st.session_state.documents_processed:
    # Display welcome message and instructions
    st.markdown("""
    ## Welcome to Mysterious Document Analyzer! 👋

    This application helps you analyze your documents.

    ### Getting Started

    1. Choose whether to process new documents or load an existing database in the sidebar.
    2. Follow the instructions to either preprocess your data or load the existing setup.
    3. Once processed/loaded, you can ask questions in the main area.
    """)

# Footer
st.markdown("---")
st.markdown("Mysterious Document Analyzer © 2025")


# st.title('Ask watsonx')
# # Setup a session state message variable to hold all the old messages 
# if 'messages' not in st.session_state:
#     st.session_state.messages = []
# # Display all the historical messages
# for message in st.session_state.messages:
#     st.chat_message(message['role']).markdown(message['content'])
# # Build a prompt input template to display the prompts 
# prompt = st.chat_input('Pass Your Prompt here')
