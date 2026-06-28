from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents: List[Document]) -> List[Document]:
    """Split long page texts into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        # Chunk size controls the maximum number of characters in each chunk.
        chunk_size=800,
        # Overlap keeps nearby context between chunks so answers do not lose meaning.
        chunk_overlap=150,
        length_function=len,
    )

    # The splitter keeps document metadata, so each chunk still knows its source PDF and page.
    return splitter.split_documents(documents)
