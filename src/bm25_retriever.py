import re
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """Convert text into lowercase word tokens for BM25 keyword matching."""
    # BM25 works with exact terms, so we split text into simple word tokens.
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """Simple BM25 retriever built from the same chunks used by FAISS."""

    def __init__(self, chunks: List[Document]) -> None:
        self.chunks = chunks
        self.tokenized_chunks = [tokenize(chunk.page_content) for chunk in chunks]

        # Vector search uses embeddings to find semantic similarity.
        # BM25 does not use embeddings. It scores chunks using keyword overlap,
        # term frequency, and how rare a word is across all chunks.
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def retrieve_with_scores(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Return the top matching chunks with BM25 scores."""
        tokenized_query = tokenize(query)
        query_terms = set(tokenized_query)

        # BM25 gives one score per chunk. Higher scores mean stronger keyword match.
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indexes = sorted(
            range(len(self.chunks)),
            key=lambda index: (
                scores[index],
                len(query_terms.intersection(self.tokenized_chunks[index])),
            ),
            reverse=True,
        )

        # Return the original Document objects, so metadata is preserved.
        return [(self.chunks[index], float(scores[index])) for index in ranked_indexes[:top_k]]

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Return only the top matching chunks with their original metadata."""
        return [doc for doc, _score in self.retrieve_with_scores(query, top_k)]


def create_bm25_retriever(chunks: List[Document]) -> BM25Retriever:
    """Create a BM25 retriever from LangChain document chunks."""
    return BM25Retriever(chunks)
