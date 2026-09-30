import os

from dotenv import load_dotenv


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-rag"
)

EMBEDDING_MODEL = "text-embedding-3-small"

LLM_MODEL = "gpt-4.1-mini"

CHUNK_SIZE = 900

CHUNK_OVERLAP = 150

TOP_K = 5


if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing.")