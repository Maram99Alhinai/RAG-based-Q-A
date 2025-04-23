# about the app
AI Engineer - NLP, create A Built an app with the following steps :
* Processing a mix of .docx, .pdf, and tabular formats (.csv, .xlsx, etc.)
* Chunking and organizing the documents.
* Building a vector database using nomic embeddings.
* Developing a RAG-based Q&A system with a local LLM (e.g., LLaMA).
* Translating and summarizing documents in various language

To analyze a collection of mysterious documents left behind by a person.

mysterious_document_analyzer/
├── data/              #Data
│   ├── Raw  # Stores the input documents (.docx, .pdf, .csv, .xlsx)
│   ├── vector_db         # Stores the Nomic Atlas vector database
├── models/            # Stores the local LLM (e.g., LLaMA)
├── utils/
│   ├── __init__.py 
│   ├── document_processing.py  # Handles document loading, chunking
│   ├── vector_db_utils.py    # Handles vector database operations
│   ├── llm_utils.py          # Handles LLM interactions
│   ├── translation_utils.py # Handles translation and summarization
├── app.py             # Streamlit application
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation

## pipline 

Build the document loading → chunking → vectorizing → retrieval → answering pipeline first.


## Prereqest 
- Python 3.11.4
-  Nomic AI account for the project and token 



## steps 
- python -m venv venv  
- venv\Scripts\activate
- pip install -r requirements.txt 
- mkdir -p models
- cd models
- git lfs install
- git clone git@hf.co:meta-llama/Llama-2-7b-chat-hf
- git clone git@hf.co:Helsinki-NLP/opus-mt-en-mt
- git clone git@hf.co:microsoft/Phi-3-mini-4k-instruct
- cd ..
- streamlit run app.py



# changes to make 
make the user select the data folder
show message when each step is done 
-save and show the chat history 
