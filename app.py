import streamlit as st

from src.bm25_retriever import create_bm25_retriever
from src.hybrid_search import hybrid_search
from src.pdf_loader import load_pdf_documents, save_uploaded_pdfs
from src.qa_chain import answer_question
from src.text_splitter import split_documents
from src.vector_store import create_vector_store, load_vector_store


st.set_page_config(page_title="Multi-PDF RAG QA", layout="wide")

st.title("Multi-PDF RAG Question Answering")
st.caption("Upload PDFs, build a retrieval index, and chat with your documents.")


# Store the vector database in session state so it remains available
# while the Streamlit app is running.
if "vector_store" not in st.session_state:
    try:
        # If a FAISS index was created earlier, load it automatically on app start.
        st.session_state.vector_store = load_vector_store()
    except Exception as error:
        st.session_state.vector_store = None
        st.warning(f"Could not load existing FAISS index: {error}")

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None

if "document_count" not in st.session_state:
    st.session_state.document_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "upload_status" not in st.session_state:
    st.session_state.upload_status = "No PDFs processed yet."

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


with st.sidebar:
    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    build_index = st.button("Process PDFs", type="primary")
    clear_chat = st.button("Clear Chat")

    if clear_chat:
        # Clear only the conversation, not the uploaded documents or indexes.
        st.session_state.chat_messages = []

    st.divider()
    st.subheader("Upload Status")
    st.write(st.session_state.upload_status)

    col1, col2 = st.columns(2)
    col1.metric("Documents", st.session_state.document_count)
    col2.metric("Chunks", st.session_state.chunk_count)

    if st.session_state.processed_files:
        st.subheader("Processed Files")
        for file_name in st.session_state.processed_files:
            st.write(f"- {file_name}")


if build_index:
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        st.session_state.upload_status = "Waiting for PDF upload."
    else:
        with st.spinner("Extracting text from PDFs..."):
            # Step 1: Save uploaded PDFs so they are visible in the project folder.
            saved_file_paths = save_uploaded_pdfs(uploaded_files)

            # Step 2: Read all uploaded PDFs and convert their pages into documents.
            documents, pdf_errors = load_pdf_documents(uploaded_files)

        if pdf_errors:
            st.warning("Some PDFs or pages could not be processed.")
            with st.expander("PDF processing errors"):
                for error in pdf_errors:
                    st.write(f"- {error}")

        if not documents:
            st.error("No readable text was found in the uploaded PDFs.")
            st.session_state.upload_status = "No readable text found."
        else:
            with st.spinner("Splitting text into chunks..."):
                # Step 3: Split long PDF text into smaller chunks for better retrieval.
                chunks = split_documents(documents)

            with st.spinner("Creating FAISS vector store..."):
                # Step 4: Convert chunks into embeddings, store them in FAISS,
                # and save the FAISS index to data/faiss_index.
                st.session_state.vector_store = create_vector_store(chunks)
                st.session_state.processed_files = [file.name for file in uploaded_files]
                st.session_state.document_count = len(documents)
                st.session_state.chunk_count = len(chunks)

            with st.spinner("Creating BM25 keyword retriever..."):
                # Step 5: Build BM25 from the same chunks used by FAISS.
                # BM25 is keyword based, while FAISS vector search is meaning based.
                st.session_state.bm25_retriever = create_bm25_retriever(chunks)

            st.success(f"Processed {len(uploaded_files)} PDF file(s) into {len(chunks)} chunks.")
            st.info("Saved uploaded files in data/uploaded_pdfs.")
            st.info("Saved FAISS index in data/faiss_index.")
            st.session_state.upload_status = "Ready for questions."

st.subheader("Chat")

if not st.session_state.chat_messages:
    st.info("Upload and process PDFs, then ask a question below.")

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant":
            citations = message.get("citations", [])
            if citations:
                st.markdown("**Sources**")
                for citation in citations:
                    st.markdown(f"- **{citation['filename']}** Page {citation['page_number']}")

            source_chunks = message.get("sources", [])
            if source_chunks:
                with st.expander("Source Chunks"):
                    for index, source in enumerate(source_chunks, start=1):
                        st.markdown(f"**Source {index}: {source['source']} - Page {source['page']}**")
                        st.write(source["content"])

            hybrid_results = message.get("hybrid_results", [])
            if hybrid_results:
                with st.expander("Hybrid Retrieval Scores"):
                    for index, item in enumerate(hybrid_results, start=1):
                        st.markdown(f"**Hybrid {index}: {item['filename']} - Page {item['page_number']}**")
                        st.write(f"Combined score: {item['combined_score']:.3f}")
                        st.write(f"FAISS score: {item['faiss_score']:.3f}")
                        st.write(f"BM25 score: {item['bm25_score']:.3f}")


question = st.chat_input("Ask a question about your uploaded PDFs")

if question:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process PDFs first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Searching documents and generating answer..."):
            if st.session_state.bm25_retriever is not None:
                # Hybrid retrieval gets top 5 chunks from FAISS and top 5 from BM25,
                # merges duplicates, and ranks the chunks by a combined score.
                hybrid_results = hybrid_search(
                    st.session_state.vector_store,
                    st.session_state.bm25_retriever,
                    question,
                    top_k=5,
                )
                retrieved_docs = [item["document"] for item in hybrid_results]
            else:
                hybrid_results = []
                retrieved_docs = None

            # Step 6: Ask the OpenRouter model to answer using retrieved chunks.
            result = answer_question(st.session_state.vector_store, question, retrieved_docs)

        display_hybrid_results = []
        if hybrid_results:
            for item in hybrid_results:
                doc = item["document"]
                display_hybrid_results.append(
                    {
                        "filename": doc.metadata.get("filename", doc.metadata.get("source", "Unknown source")),
                        "page_number": doc.metadata.get("page_number", doc.metadata.get("page", "Unknown page")),
                        "combined_score": item["combined_score"],
                        "faiss_score": item["faiss_score"],
                        "bm25_score": item["bm25_score"],
                    }
                )

        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"],
            "sources": result["sources"],
            "hybrid_results": display_hybrid_results,
        }
        st.session_state.chat_messages.append(assistant_message)

        with st.chat_message("assistant"):
            st.write(assistant_message["content"])

            st.markdown("**Sources**")
            if assistant_message["citations"]:
                for citation in assistant_message["citations"]:
                    st.markdown(f"- **{citation['filename']}** Page {citation['page_number']}")
            else:
                st.write("No sources found.")

            with st.expander("Source Chunks"):
                for index, source in enumerate(assistant_message["sources"], start=1):
                    st.markdown(f"**Source {index}: {source['source']} - Page {source['page']}**")
                    st.write(source["content"])

            if assistant_message["hybrid_results"]:
                with st.expander("Hybrid Retrieval Scores"):
                    for index, item in enumerate(assistant_message["hybrid_results"], start=1):
                        st.markdown(f"**Hybrid {index}: {item['filename']} - Page {item['page_number']}**")
                        st.write(f"Combined score: {item['combined_score']:.3f}")
                        st.write(f"FAISS score: {item['faiss_score']:.3f}")
                        st.write(f"BM25 score: {item['bm25_score']:.3f}")
