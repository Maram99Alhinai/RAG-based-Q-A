import streamlit as st
from utils.document_processing import load_documents, chunk_documents
from utils.vector_db_utils import create_vector_db, retrieve_relevant_documents
from utils.llm_utils import setup_llm, create_qa_chain
from utils.translation_utils import translate_text, summarize_text
import os


st.write("Hello World")

# # Configuration
# DATA_PATH = "data"
# VECTOR_DB_PATH = "vector_db"
# LLM_MODEL_PATH = "models/llama-2-7b-chat.ggmlv3.q4_0.bin" #replace with your model
# ATLAS_TOKEN = os.environ.get("ATLAS_TOKEN") #set your atlas token as an env variable
# ATLAS_PROJECT_NAME = "mysterious_documents"
# TRANSLATION_MODEL_PATH = "models/opus-mt-en-fr" #replace with your translation model.

# st.title("Mysterious Document Analyzer")

# if st.button("Process Documents"):
#     with st.spinner("Processing documents..."):
#         documents = load_documents(DATA_PATH)
#         chunks = chunk_documents(documents)
#         vector_db = create_vector_db(chunks, ATLAS_TOKEN, ATLAS_PROJECT_NAME)
#         st.session_state.vector_db = vector_db
#         llm = setup_llm(LLM_MODEL_PATH)
#         st.session_state.qa_chain = create_qa_chain(llm, vector_db)
#         st.success("Documents processed!")

# if "qa_chain" in st.session_state:
#     query = st.text_input("Ask a question about the documents:")
#     if query:
#         with st.spinner("Answering..."):
#             answer = st.session_state.qa_chain({"query": query})["result"]
#             st.write("Answer:", answer)
#             relevant_docs = retrieve_relevant_documents(st.session_state.vector_db, query)
#             st.write("Relevant Documents:", relevant_docs)

#     translate_query = st.text_input("Translate some text:")
#     if translate_query:
#         translated_text = translate_text(translate_query, "en", "fr", TRANSLATION_MODEL_PATH)
#         st.write("Translated Text:", translated_text)

#     summarize_query = st.text_area("Summarize some text:")
#     if summarize_query:
#         llm = setup_llm(LLM_MODEL_PATH)
#         summary = summarize_text(llm, summarize_query)
#         st.write("Summary:", summary)