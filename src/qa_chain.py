from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, validate_openrouter_key


def get_llm() -> ChatOpenAI:
    """Create an OpenRouter chat model using the OpenAI-compatible API."""
    validate_openrouter_key()

    return ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=OPENROUTER_MODEL,
        temperature=0.2,
    )


def build_context(retrieved_docs) -> str:
    """Combine retrieved document chunks into a single context string."""
    context_parts = []

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        context_parts.append(f"Source: {source}, Page: {page}\n{doc.page_content}")

    return "\n\n".join(context_parts)


def build_citations(retrieved_docs) -> List[Dict[str, Any]]:
    """Create a clean, duplicate-free citation list from retrieved chunks."""
    citations = []
    seen_sources = set()

    for doc in retrieved_docs:
        filename = doc.metadata.get("filename", doc.metadata.get("source", "Unknown source"))
        page_number = doc.metadata.get("page_number", doc.metadata.get("page", "Unknown page"))
        citation_key = (filename, page_number)

        # Multiple chunks can come from the same PDF page, so show that page only once.
        if citation_key in seen_sources:
            continue

        seen_sources.add(citation_key)
        citations.append(
            {
                "filename": filename,
                "page_number": page_number,
                "label": f"{filename} Page {page_number}",
            }
        )

    return citations


def answer_question(vector_store, question: str, retrieved_docs=None) -> Dict[str, Any]:
    """Answer the question using retrieved chunks and OpenRouter."""
    if retrieved_docs is None:
        # Fallback path: use FAISS only if no hybrid results are provided.
        retrieved_docs = vector_store.similarity_search(question, k=5)

    # Put retrieved chunks into the prompt as context.
    context = build_context(retrieved_docs)

    prompt = f"""
You are a helpful assistant for a PDF question-answering system.
Answer the question using only the context below.
If the answer is not present in the context, say that you could not find it in the uploaded PDFs.
At the end of the answer, mention that the sources are listed below the answer.

Context:
{context}

Question:
{question}

Answer:
"""

    # Ask the OpenRouter model to generate the final answer.
    llm = get_llm()
    response = llm.invoke(prompt)

    sources = []
    for doc in retrieved_docs:
        sources.append(
            {
                "source": doc.metadata.get("source", "Unknown source"),
                "page": doc.metadata.get("page", "Unknown page"),
                "content": doc.page_content,
            }
        )

    return {
        "answer": response.content,
        "sources": sources,
        "citations": build_citations(retrieved_docs),
    }
