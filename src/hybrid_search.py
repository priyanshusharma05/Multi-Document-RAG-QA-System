import hashlib
from typing import Any, Dict, List

from langchain_core.documents import Document


def document_key(document: Document) -> str:
    """Create a stable key so the same chunk is not returned twice."""
    filename = document.metadata.get("filename", document.metadata.get("source", ""))
    page_number = document.metadata.get("page_number", document.metadata.get("page", ""))
    content_hash = hashlib.md5(document.page_content.encode("utf-8")).hexdigest()
    return f"{filename}:{page_number}:{content_hash}"


def normalize_scores(raw_scores: List[float], higher_is_better: bool = True) -> List[float]:
    """Convert scores to a 0 to 1 range so FAISS and BM25 can be combined."""
    if not raw_scores:
        return []

    if not higher_is_better:
        # FAISS distance scores are usually lower when the match is better.
        raw_scores = [-score for score in raw_scores]

    minimum = min(raw_scores)
    maximum = max(raw_scores)

    if minimum == maximum:
        return [1.0 for _score in raw_scores]

    return [(score - minimum) / (maximum - minimum) for score in raw_scores]


def hybrid_search(vector_store, bm25_retriever, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve chunks using FAISS and BM25, then merge them into one ranked list."""
    # Step 1: FAISS vector search finds chunks with similar meaning.
    # It can match related ideas even when the exact words are different.
    faiss_results = vector_store.similarity_search_with_score(query, k=top_k)

    # Step 2: BM25 keyword search finds chunks with strong word overlap.
    # It is useful when exact terms, names, definitions, or formulas matter.
    bm25_results = bm25_retriever.retrieve_with_scores(query, top_k=top_k)

    faiss_scores = normalize_scores([float(score) for _doc, score in faiss_results], higher_is_better=False)
    bm25_scores = normalize_scores([float(score) for _doc, score in bm25_results], higher_is_better=True)

    merged_results: Dict[str, Dict[str, Any]] = {}

    # Step 3: Add FAISS results to the merged result dictionary.
    for (document, _raw_score), score in zip(faiss_results, faiss_scores):
        key = document_key(document)
        merged_results[key] = {
            "document": document,
            "faiss_score": score,
            "bm25_score": 0.0,
            "combined_score": score,
            "metadata": document.metadata,
        }

    # Step 4: Add BM25 results. If the same chunk already exists, update its score.
    for (document, _raw_score), score in zip(bm25_results, bm25_scores):
        key = document_key(document)

        if key not in merged_results:
            merged_results[key] = {
                "document": document,
                "faiss_score": 0.0,
                "bm25_score": score,
                "combined_score": score,
                "metadata": document.metadata,
            }
        else:
            merged_results[key]["bm25_score"] = score
            merged_results[key]["combined_score"] = merged_results[key]["faiss_score"] + score

    # Step 5: Rank by the simple combined score and return the best chunks.
    ranked_results = sorted(
        merged_results.values(),
        key=lambda result: result["combined_score"],
        reverse=True,
    )

    return ranked_results[:top_k]
