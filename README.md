# 🤖 Agentic AI RAG

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about an **Agentic AI ebook** using OpenAI embeddings, Pinecone vector search, GPT-4.1-mini, and a Streamlit chat interface.

The application follows this pipeline:

```text
PDF Document
     │
     ▼
Text Extraction
     │
     ▼
Text Chunking
     │
     ▼
OpenAI Embeddings
text-embedding-3-small
     │
     ▼
1536-dimensional vectors
     │
     ▼
Pinecone Vector Database
     │
     ▼
Semantic Retrieval
     │
     ▼
Relevant Context
     │
     ▼
GPT-4.1-mini
     │
     ▼
Final Answer
     │
     ▼
Streamlit UI
```

---

## 📌 Features

* PDF document ingestion
* Page-level text extraction
* Recursive text chunking
* OpenAI `text-embedding-3-small` embeddings
* 1536-dimensional vector embeddings
* Pinecone vector database
* Semantic similarity search
* Top-K document retrieval
* GPT-4.1-mini response generation
* Source/page references
* Streamlit chat interface
* Conversation history during the current session
* Clear chat functionality
* Configurable chunk size and overlap
* Configurable retrieval count

---

# 🏗️ Project Structure

```text
D:\agentic-ai-rag
│
├── .env
├── .gitignore
├── README.md
├── config.py
├── app.py
│
├── data
│   └── Ebook-Agentic-AI.pdf
│
├── src
│   ├── ingest.py
│   ├── retrieve.py
│   └── rag.py
│
└── .venv
```

### File Responsibilities

| File                        | Purpose                                                                      |
| --------------------------- | ---------------------------------------------------------------------------- |
| `app.py`                    | Streamlit web application                                                    |
| `config.py`                 | Environment variables and application configuration                          |
| `src/ingest.py`             | Extracts PDF text, creates chunks, generates embeddings, and uploads vectors |
| `src/retrieve.py`           | Tests semantic retrieval from Pinecone                                       |
| `src/rag.py`                | Command-line RAG chatbot                                                     |
| `.env`                      | API keys and environment configuration                                       |
| `data/Ebook-Agentic-AI.pdf` | Source document                                                              |
| `README.md`                 | Project documentation                                                        |

---

# ⚙️ Technologies Used

* Python
* Streamlit
* OpenAI
* LangChain
* Pinecone
* PyPDF
* Recursive Character Text Splitter
* python-dotenv

---

# 🔑 Models

## Embedding Model

```text
text-embedding-3-small
```

Embedding dimension:

```text
1536
```

## LLM

```text
gpt-4.1-mini
```

---

# 🗄️ Pinecone Configuration

The application uses:

```text
Index:
agentic-ai-rag-1536
```

Vector dimension:

```text
1536
```

Metric:

```text
cosine
```

Namespace:

```text
agentic-ai
```

---

# 🐍 1. Create the Virtual Environment

From the project directory:

```powershell
cd D:\agentic-ai-rag
```

Create the virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see:

```text
(.venv) PS D:\agentic-ai-rag>
```

---

# 📦 2. Install Dependencies

Install the required packages:

```powershell
pip install openai langchain langchain-openai langchain-text-splitters pypdf pinecone python-dotenv streamlit
```

You can verify Streamlit:

```powershell
streamlit --version
```

---

# 🔐 3. Configure Environment Variables

Create:

```text
D:\agentic-ai-rag\.env
```

Add:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-rag-1536
```

Replace:

```text
your_openai_api_key
```

and:

```text
your_pinecone_api_key
```

with your actual API keys.

Do not commit `.env` to GitHub.

---

# ⚙️ 4. Configuration

The project uses:

```text
config.py
```

Current configuration:

```python
EMBEDDING_MODEL = "text-embedding-3-small"

LLM_MODEL = "gpt-4.1-mini"

CHUNK_SIZE = 900

CHUNK_OVERLAP = 150

TOP_K = 5
```

### Chunk Configuration

```text
Chunk size:
900 characters

Chunk overlap:
150 characters
```

### Retrieval Configuration

```text
Top K:
5
```

This means the system retrieves the five most relevant chunks from Pinecone for each question.

---

# 📄 5. Add the PDF

Place the source document here:

```text
D:\agentic-ai-rag\data\Ebook-Agentic-AI.pdf
```

The ingestion script automatically looks for:

```text
data/Ebook-Agentic-AI.pdf
```

---

# 📥 6. Ingest the PDF

Run:

```powershell
python src/ingest.py
```

The ingestion process performs:

```text
PDF
 ↓
Extract pages
 ↓
Create text chunks
 ↓
Generate embeddings
 ↓
Upload vectors to Pinecone
```

Expected output:

```text
AGENTIC AI RAG - DOCUMENT INGESTION

PDF: D:\agentic-ai-rag\data\Ebook-Agentic-AI.pdf
Embedding model: text-embedding-3-small
Embedding dimension: 1536
Pinecone index: agentic-ai-rag-1536
Namespace: agentic-ai

Extracted 59 pages.
Created 128 chunks.

Creating embeddings for 1-50 of 128...
Uploaded 50/128

Creating embeddings for 51-100 of 128...
Uploaded 100/128

Creating embeddings for 101-128 of 128...
Uploaded 128/128

Ingestion completed successfully.
```

The exact number of pages/chunks can change if the PDF is replaced.

---

# 🔎 7. Test Semantic Retrieval

After successful ingestion, run:

```powershell
python src/retrieve.py
```

The application will ask:

```text
Enter your question:
```

Example:

```text
What is Agentic AI?
```

The system will:

```text
Question
   ↓
Generate query embedding
   ↓
Search Pinecone
   ↓
Retrieve Top 5 chunks
   ↓
Display matching content
```

The output should contain:

```text
Result 1

Score: ...
Page: ...
Chunk: ...

Relevant text...
```

---

# 🧠 8. Run the Command-Line RAG Application

Run:

```powershell
python src/rag.py
```

You should see:

```text
============================================================
AGENTIC AI RAG CHATBOT
============================================================

You:
```

Ask a question:

```text
You: What is Agentic AI?
```

The system performs:

```text
Question
    ↓
Embedding
    ↓
Pinecone Retrieval
    ↓
Relevant Context
    ↓
GPT-4.1-mini
    ↓
Answer
```

To exit:

```text
exit
```

or:

```text
quit
```

---

# 🖥️ 9. Run the Streamlit Application

The main user interface is the Streamlit application.

Run:

```powershell
python -m streamlit run app.py
```

Streamlit will start a local server.

Typically the application will be available at:

```text
http://localhost:8501
```

Open the address in your browser if it does not open automatically.

---

# 💬 10. Using the Streamlit Application

The interface provides a chat-based RAG experience.

Example:

```text
User:
What is Agentic AI?

Assistant:
[Generated answer based on the retrieved ebook content]

📚 Sources

Source 1
Page: 5
Relevance score: 0.82

[Retrieved text]
```

The application displays retrieved sources so that the user can inspect which sections of the ebook were used.

---

# 🔄 RAG Workflow

The complete application works as follows:

```text
                    USER
                     │
                     ▼
             Streamlit Chat UI
                     │
                     ▼
               User Question
                     │
                     ▼
          OpenAI Embedding Model
        text-embedding-3-small
                     │
                     ▼
              1536-D Vector
                     │
                     ▼
               Pinecone
                     │
             Similarity Search
                     │
                     ▼
                Top 5 Chunks
                     │
                     ▼
              Retrieved Context
                     │
                     ▼
                GPT-4.1-mini
                     │
                     ▼
              Generated Answer
                     │
                     ▼
             Streamlit Interface
```

---

# 🧩 Why Pinecone Uses 1536 Dimensions

The embedding model used by this project is:

```text
text-embedding-3-small
```

The generated embeddings contain:

```text
1536 dimensions
```

Therefore the Pinecone index must also have:

```text
dimension = 1536
```

The index configuration must match the embedding dimension.

---

# ⚠️ Previous Dimension Error

An earlier Pinecone index was configured with:

```text
Dimension: 1024
```

while the OpenAI embedding model generated:

```text
1536 dimensions
```

This caused:

```text
Vector dimension 1536 does not match the dimension of the index 1024
```

The solution was to use a new Pinecone index:

```text
agentic-ai-rag-1536
```

with:

```text
Dimension: 1536
```

The current project should use:

```env
PINECONE_INDEX_NAME=agentic-ai-rag-1536
```

---

# 🧪 Testing Checklist

After setup, test the system in this order.

## Test 1 — Environment

```powershell
python -c "import config; print(config.__file__)"
```

Expected:

```text
D:\agentic-ai-rag\config.py
```

---

## Test 2 — Ingestion

```powershell
python src/ingest.py
```

Expected:

```text
Ingestion completed successfully.
```

---

## Test 3 — Retrieval

```powershell
python src/retrieve.py
```

Ask:

```text
What is Agentic AI?
```

Verify that relevant chunks are returned.

---

## Test 4 — RAG

```powershell
python src/rag.py
```

Ask a question about the ebook.

Verify that GPT generates an answer based on retrieved context.

---

## Test 5 — Streamlit

```powershell
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

Ask questions through the web interface.

---

# 🧯 Troubleshooting

## `ModuleNotFoundError: No module named 'config'`

Make sure the project has:

```text
D:\agentic-ai-rag\config.py
```

and not only:

```text
D:\agentic-ai-rag\src\config.py
```

Correct structure:

```text
D:\agentic-ai-rag
├── app.py
├── config.py
└── src
    ├── ingest.py
    ├── retrieve.py
    └── rag.py
```

Test:

```powershell
Test-Path .\config.py
```

Expected:

```text
True
```

---

## `OPENAI_API_KEY is missing`

Check `.env`:

```text
D:\agentic-ai-rag\.env
```

Make sure:

```env
OPENAI_API_KEY=your_key
```

is present.

Restart the terminal after changing environment configuration if necessary.

---

## `PINECONE_API_KEY is missing`

Make sure `.env` contains:

```env
PINECONE_API_KEY=your_key
```

---

## Vector Dimension Error

If you see:

```text
Vector dimension 1536 does not match the dimension of the index 1024
```

verify that `.env` points to:

```env
PINECONE_INDEX_NAME=agentic-ai-rag-1536
```

and that the Pinecone index has:

```text
Dimension: 1536
```

---

## Streamlit Command Not Found

Instead of:

```powershell
streamlit run app.py
```

use:

```powershell
python -m streamlit run app.py
```

---

## Pinecone Index Not Found

Check:

```env
PINECONE_INDEX_NAME=agentic-ai-rag-1536
```

Then verify that the index exists in your Pinecone project.

---

# 🔒 Security

Never commit API keys to GitHub.

Your `.gitignore` should contain:

```gitignore
.venv/
.env
__pycache__/
*.pyc
.streamlit/secrets.toml
```

Never place API keys directly inside:

```text
app.py
config.py
ingest.py
retrieve.py
rag.py
```

Use `.env` instead.

---

# 🚀 Running the Project From Scratch

After cloning or copying the project:

```powershell
cd D:\agentic-ai-rag
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install openai langchain langchain-openai langchain-text-splitters pypdf pinecone python-dotenv streamlit
```

Configure `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-rag-1536
```

Place the PDF:

```text
data/Ebook-Agentic-AI.pdf
```

Run ingestion:

```powershell
python src/ingest.py
```

Test retrieval:

```powershell
python src/retrieve.py
```

Run the RAG CLI:

```powershell
python src/rag.py
```

Run the web application:

```powershell
python -m streamlit run app.py
```

---

# 📈 Future Improvements

Potential improvements for the next versions include:

* Conversation-aware retrieval
* Chat history summarization
* Hybrid search
* Metadata filtering
* Reranking retrieved chunks
* Better citation handling
* Streaming LLM responses
* Document upload through Streamlit
* Multiple PDF support
* Multiple namespaces
* Authentication
* Persistent chat history
* Evaluation datasets
* Retrieval quality metrics
* RAG evaluation with precision/recall
* LangGraph-based agentic workflows
* Query rewriting
* Context compression
* Multi-step agentic retrieval
* Web search fallback
* Source confidence scoring
* Deployment to Streamlit Community Cloud or another hosting platform

---

# 🎯 Current Project Status

```text
✅ PDF extraction
✅ Page-level text extraction
✅ Text chunking
✅ OpenAI embeddings
✅ 1536-dimensional embeddings
✅ Pinecone vector storage
✅ Semantic retrieval
✅ GPT-based answer generation
✅ Source metadata
✅ CLI RAG application
✅ Streamlit chat interface
```

The core RAG pipeline is:

```text
PDF
 ↓
Chunking
 ↓
Embedding
 ↓
Pinecone
 ↓
Retrieval
 ↓
Context
 ↓
GPT-4.1-mini
 ↓
Streamlit
```

---

# 👨‍💻 Project

**Agentic AI RAG**

A document-grounded AI question-answering system built with:

```text
Python
OpenAI
LangChain
Pinecone
Streamlit
```

The system is designed to answer questions using information retrieved from the provided Agentic AI ebook rather than relying solely on the language model's general knowledge.
#   a g e n t i c - a i - r a g 
 
 