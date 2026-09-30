from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ============================================================
# PDF PATH
# ============================================================

PDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "Ebook-Agentic-AI.pdf"
)


# ============================================================
# PINECONE CONFIGURATION
# ============================================================

# text-embedding-3-small produces 1536-dimensional vectors
EMBEDDING_DIMENSION = 1536

NAMESPACE = "agentic-ai"


# ============================================================
# EXTRACT PDF PAGES
# ============================================================

def extract_pages():
    """Extract text page-by-page from the PDF."""

    reader = PdfReader(str(PDF_PATH))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        text = text.strip()

        if not text:
            continue

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    print(f"Extracted {len(pages)} pages.")

    return pages


# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks(pages):
    """Split each page into smaller chunks while preserving page metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for page_data in pages:

        page_chunks = splitter.split_text(
            page_data["text"]
        )

        for chunk_number, chunk_text in enumerate(
            page_chunks,
            start=1,
        ):

            if not chunk_text.strip():
                continue

            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "page": page_data["page"],
                    "chunk": chunk_number,
                }
            )

    print(f"Created {len(chunks)} chunks.")

    return chunks


# ============================================================
# GET / CREATE PINECONE INDEX
# ============================================================

def get_pinecone_index():

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    existing_indexes = pc.list_indexes().names()

    # --------------------------------------------------------
    # Create index if it doesn't exist
    # --------------------------------------------------------

    if PINECONE_INDEX_NAME not in existing_indexes:

        print(
            f"Creating Pinecone index "
            f"'{PINECONE_INDEX_NAME}'..."
        )

        print(
            f"Index dimension: "
            f"{EMBEDDING_DIMENSION}"
        )

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        print("Pinecone index created successfully.")

    else:

        print(
            f"Using existing Pinecone index: "
            f"{PINECONE_INDEX_NAME}"
        )

    # --------------------------------------------------------
    # Connect to index
    # --------------------------------------------------------

    index = pc.Index(
        PINECONE_INDEX_NAME
    )

    return index


# ============================================================
# UPLOAD TO PINECONE
# ============================================================

def upload_to_pinecone(chunks):

    print()
    print("Initializing OpenAI embeddings...")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Expected embedding dimension: "
        f"{EMBEDDING_DIMENSION}"
    )

    index = get_pinecone_index()

    batch_size = 50

    total_chunks = len(chunks)

    # --------------------------------------------------------
    # Process chunks in batches
    # --------------------------------------------------------

    for start in range(
        0,
        total_chunks,
        batch_size,
    ):

        batch = chunks[
            start:start + batch_size
        ]

        texts = [
            item["text"]
            for item in batch
        ]

        print()
        print(
            f"Creating embeddings for "
            f"{start + 1}-{min(start + batch_size, total_chunks)} "
            f"of {total_chunks}..."
        )

        vectors = embeddings.embed_documents(
            texts
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if vectors:

            actual_dimension = len(
                vectors[0]
            )

            if actual_dimension != EMBEDDING_DIMENSION:

                raise ValueError(
                    f"Embedding dimension mismatch! "
                    f"Expected {EMBEDDING_DIMENSION}, "
                    f"but received {actual_dimension}."
                )

        # ----------------------------------------------------
        # Create Pinecone records
        # ----------------------------------------------------

        records = []

        for i, (
            item,
            vector,
        ) in enumerate(
            zip(batch, vectors)
        ):

            vector_id = (
                f"page-{item['page']}-"
                f"chunk-{item['chunk']}-"
                f"{start + i}"
            )

            records.append(
                {
                    "id": vector_id,

                    "values": vector,

                    "metadata": {
                        "text": item["text"],
                        "page": item["page"],
                        "chunk": item["chunk"],
                        "source": "Agentic AI Ebook",
                    },
                }
            )

        # ----------------------------------------------------
        # Upload batch
        # ----------------------------------------------------

        index.upsert(
            vectors=records,
            namespace=NAMESPACE,
        )

        uploaded = min(
            start + batch_size,
            total_chunks,
        )

        print(
            f"Uploaded "
            f"{uploaded}/{total_chunks}"
        )

    print()
    print("=" * 60)
    print("Ingestion completed successfully.")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check PDF
    # --------------------------------------------------------

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    print("=" * 60)
    print("AGENTIC AI RAG - DOCUMENT INGESTION")
    print("=" * 60)

    print()
    print(
        f"PDF: {PDF_PATH}"
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    print(
        f"Embedding dimension: {EMBEDDING_DIMENSION}"
    )

    print(
        f"Pinecone index: {PINECONE_INDEX_NAME}"
    )

    print(
        f"Namespace: {NAMESPACE}"
    )

    print()

    # --------------------------------------------------------
    # Extract PDF
    # --------------------------------------------------------

    pages = extract_pages()

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = create_chunks(
        pages
    )

    # --------------------------------------------------------
    # Upload vectors
    # --------------------------------------------------------

    upload_to_pinecone(
        chunks
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()