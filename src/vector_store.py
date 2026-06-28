import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# This project uses PyTorch sentence-transformer embeddings.
# These flags stop Transformers from trying to load TensorFlow/Keras,
# which can cause avoidable errors on systems that have Keras 3 installed.
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


# Folder where FAISS writes its index files.
INDEX_DIR = Path("data/faiss_index")
BACKEND_FILE = INDEX_DIR / "embedding_backend.txt"

# Sentence-transformer model used to generate local embeddings.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load .env so the embedding backend can be changed without editing code.
load_dotenv()


def get_embedding_backend() -> str:
    """Read which embedding backend should be used for this project."""
    return os.getenv("EMBEDDING_BACKEND", "hashing").lower()


class LocalHashingEmbeddings(Embeddings):
    """Simple offline embeddings for systems that cannot access Hugging Face."""

    def __init__(self) -> None:
        # HashingVectorizer does not download any model.
        # It is less accurate than all-MiniLM-L6-v2, but keeps the project runnable offline.
        self.vectorizer = HashingVectorizer(
            n_features=4096,
            alternate_sign=False,
            norm=None,
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # Convert text into numeric vectors and normalize them for FAISS similarity search.
        sparse_vectors = self.vectorizer.transform(texts)
        normalized_vectors = normalize(sparse_vectors, norm="l2")
        return normalized_vectors.toarray().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create vectors for document chunks."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Create a vector for the user's question."""
        return self._embed([text])[0]


def get_embedding_model() -> Embeddings:
    """Create the local embedding model used to convert text into vectors."""
    backend = get_embedding_backend()

    if backend == "sentence-transformers":
        # This uses all-MiniLM-L6-v2 and generates embeddings locally.
        # It needs internet only the first time, when the model is downloaded.
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Default backend for your current machine because huggingface.co is not resolving.
    return LocalHashingEmbeddings()


def create_vector_store(chunks: List[Document]) -> FAISS:
    """Create a FAISS vector store from chunks and save it to disk."""
    embeddings = get_embedding_model()

    # FAISS stores vector embeddings and LangChain stores each chunk's metadata with it.
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save both the FAISS index and document metadata for future app runs.
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))
    BACKEND_FILE.write_text(get_embedding_backend(), encoding="utf-8")

    return vector_store


def load_vector_store() -> FAISS | None:
    """Load an existing FAISS index from disk if it has already been created."""
    index_file = INDEX_DIR / "index.faiss"
    metadata_file = INDEX_DIR / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        return None

    # Do not load an index created with a different embedding backend.
    # Different embedding models produce different vector sizes.
    if not BACKEND_FILE.exists():
        return None

    saved_backend = BACKEND_FILE.read_text(encoding="utf-8").strip()
    if saved_backend != get_embedding_backend():
        return None

    embeddings = get_embedding_model()

    # LangChain stores metadata in a pickle file.
    # allow_dangerous_deserialization is required when loading a local index
    # that we created ourselves in this same project.
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
