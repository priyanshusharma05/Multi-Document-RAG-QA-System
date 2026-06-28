from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError


UPLOAD_DIR = Path("data/uploaded_pdfs")


def save_uploaded_pdfs(uploaded_files) -> List[str]:
    """Save uploaded PDF files so they can be seen later in VS Code."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_file_paths = []

    for uploaded_file in uploaded_files:
        file_path = UPLOAD_DIR / uploaded_file.name

        # Move the read pointer to the beginning before saving the file.
        uploaded_file.seek(0)
        file_path.write_bytes(uploaded_file.getvalue())

        # Move the pointer back again so PdfReader can read the same uploaded file.
        uploaded_file.seek(0)
        saved_file_paths.append(str(file_path))

    return saved_file_paths


def load_pdf_documents(uploaded_files) -> Tuple[List[Document], List[str]]:
    """Extract text from uploaded PDF files and return page-level LangChain documents."""
    documents = []
    errors = []

    for uploaded_file in uploaded_files:
        try:
            # Move to the beginning so PdfReader sees the complete uploaded file.
            uploaded_file.seek(0)

            # PdfReader can read Streamlit's uploaded file object directly.
            reader = PdfReader(uploaded_file)

            # Accessing pages here helps catch corrupted PDFs early.
            pages = reader.pages
        except (PdfReadError, Exception) as error:
            # If one PDF is corrupted, skip it and continue processing other PDFs.
            errors.append(f"{uploaded_file.name}: {error}")
            continue

        for page_number, page in enumerate(pages, start=1):
            try:
                # Some PDF pages may contain only images, so extract_text can return None.
                text = page.extract_text() or ""
                text = text.strip()
            except Exception as error:
                # If one page cannot be read, skip only that page.
                errors.append(f"{uploaded_file.name}, page {page_number}: {error}")
                continue

            if text:
                # Metadata helps us trace each chunk back to its original PDF and page.
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "filename": uploaded_file.name,
                            "page_number": page_number,
                            "source": uploaded_file.name,
                            "page": page_number,
                        },
                    )
                )

    return documents, errors
