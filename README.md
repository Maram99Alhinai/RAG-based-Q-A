# 📚 Mysterious Document Analyzer

A sophisticated document analysis system powered by local LLMs and RAG architecture for uncovering insights from mysterious document collections.

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

## ✨ Overview

The Mysterious Document Analyzer is an advanced NLP application designed to process, analyze, and extract information from collections of mysterious documents. The system handles multiple document formats, translates content, and provides an intelligent question-answering interface powered by local large language models.

## 🚀 Features

- **Multi-format Document Support**: Process `.docx`, `.pdf`, `.csv`, `.xlsx`, and more
- **Intelligent Chunking**: Automatically segments documents for optimal analysis
- **Vector Database**: Utilizes Nomic embeddings for efficient semantic search
- **Local LLM Integration**: Powered by LLaMA for privacy-preserving analysis
- **Language Tools**: Translate and summarize documents in multiple languages
- **Interactive Interface**: Streamlit-based UI for intuitive document exploration
- **RAG Architecture**: Combines retrieval and generation for accurate answers

## 📁 Project Structure

```
mysterious_document_analyzer/
├── data/                  # Data storage
│   ├── Raw/               # Raw input documents (.docx, .pdf, .csv, .xlsx)
│   └── vector_db/         # Nomic Atlas vector database
├── models/                # Local LLMs (LLaMA, translation, summarization)
├── utils/                 # Utility functions
│   ├── __init__.py 
│   ├── document_processing.py  # Document loading and chunking
│   ├── vector_db_utils.py      # Vector database operations
│   ├── llm_utils.py            # LLM interactions
│   ├── translation_utils.py    # Translation functionality
│   └── summarization_utils.py  # Summarization functionality
├── app.py                 # Streamlit application
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 🔄 Pipeline

The document analysis pipeline follows these steps:
1. **Document Loading**: Parse various file formats into unified text
2. **Chunking**: Segment documents into meaningful chunks
3. **Vectorizing**: Convert text to embeddings and store in Nomic Atlas
4. **Retrieval**: Find relevant document sections for queries
5. **Answer Generation**: Use LLM to generate answers based on retrieved context

## 🧰 Prerequisites

- Python 3.11.4 or higher
- Hugging Face account (for model downloads)
- 8GB+ RAM (16GB recommended)
- 20GB+ free disk space

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mysterious-document-analyzer.git
   cd mysterious-document-analyzer
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required models:
   ```bash
   mkdir -p models
   cd models
   
   # Install Git LFS
   git lfs install
   
   # Download LLaMA model
   git clone https://YOUR_USERNAME:YOUR_TOKEN@hf.co/meta-llama/Llama-2-7b-chat-hf
   
   # Download translation model
   git clone https://YOUR_USERNAME:YOUR_TOKEN@hf.co/Helsinki-NLP/opus-mt-en-mt
   
   # Download summarization model
   git clone https://YOUR_USERNAME:YOUR_TOKEN@hf.co/facebook/bart-large-cnn
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## 🖥️ Usage

1. **Setup Data Source**:
   - Choose between default directory, file upload, or custom directory
   - Process documents to create the vector database

2. **Ask Questions**:
   - Enter queries about the document collection
   - View answers with supporting evidence from source documents

3. **Translation**:
   - Translate text or entire documents between languages
   - Download translated files for offline use

## 🔮 Future Enhancements

- Custom data folder selection
- Improved chunking strategies
- Progress indicators for each processing step
- Chat history storage and visualization
- Enhanced document summarization

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for providing access to powerful language models
- The LLaMA team for their groundbreaking work on local LLMs
- The Streamlit team for their excellent framework for ML applications