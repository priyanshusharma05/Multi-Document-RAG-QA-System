# Multi-PDF RAG QA System
 Retrieval-Augmented Generation (RAG) project built with Streamlit.

The app lets users upload multiple PDF files, extracts text, splits it into chunks, stores the chunks in a FAISS vector database, and answers questions using an OpenRouter chat model.

## Features

- Multi-PDF upload
- PDF text extraction using `pypdf`
- Text chunking using LangChain text splitters
- FAISS vector store for similarity search
- Local offline embeddings using scikit-learn
- OpenRouter API for answer generation
- Simple modular folder structure
- Environment variables for API keys

## Folder Structure

```text
multi-pdf-rag/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   ├── uploaded_pdfs/
│   │   └── .gitkeep
│   └── faiss_index/
│       └── .gitkeep
└── src/
    ├── __init__.py
    ├── config.py
    ├── pdf_loader.py
    ├── text_splitter.py
    ├── vector_store.py
    └── qa_chain.py
```

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create a `.env` file from `.env.example`.

```bash
copy .env.example .env
```

4. Add your OpenRouter API key in `.env`.

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
```

## Run The App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## How It Works

1. The user uploads one or more PDF files.
2. Text is extracted page by page from each PDF.
3. Extracted text is split into smaller chunks.
4. Chunks are converted into simple local embeddings using scikit-learn.
5. FAISS stores the embeddings and finds chunks similar to the user's question.
6. The selected chunks are sent as context to OpenRouter.
7. The model generates an answer based only on the uploaded PDFs.

## Notes

- This is a starter project, not a production system.
- Uploaded PDFs are processed in memory by default.
- The FAISS index is stored in Streamlit session state while the app is running.
- The starter project uses offline local embeddings, so it does not need to download models from Hugging Face.
- You can change the OpenRouter model in `.env`.
