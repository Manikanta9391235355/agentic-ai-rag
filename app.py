import streamlit as st

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)

from pinecone import Pinecone

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
)


# ============================================================
# CONFIGURATION
# ============================================================

NAMESPACE = "agentic-ai"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Agentic AI RAG",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #777;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .source-box {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 Agentic AI RAG</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about the Agentic AI ebook using '
    'Retrieval-Augmented Generation.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.write(
        f"**Embedding model:**  \n"
        f"`{EMBEDDING_MODEL}`"
    )

    st.write(
        f"**LLM:**  \n"
        f"`{LLM_MODEL}`"
    )

    st.write(
        f"**Pinecone index:**  \n"
        f"`{PINECONE_INDEX_NAME}`"
    )

    st.write(
        f"**Namespace:**  \n"
        f"`{NAMESPACE}`"
    )

    st.write(
        f"**Top K:**  \n"
        f"`{TOP_K}`"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# PINECONE
# ============================================================

@st.cache_resource
def get_pinecone_index():

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    index = pc.Index(
        PINECONE_INDEX_NAME
    )

    return index


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )


# ============================================================
# LLM
# ============================================================

@st.cache_resource
def get_llm():

    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
    )


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    query,
):

    embeddings = get_embeddings()

    index = get_pinecone_index()

    query_vector = embeddings.embed_query(
        query
    )

    results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
        namespace=NAMESPACE,
    )

    documents = []

    for match in results.get(
        "matches",
        [],
    ):

        metadata = match.get(
            "metadata",
            {},
        )

        text = metadata.get(
            "text",
            "",
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        chunk = metadata.get(
            "chunk",
            "Unknown",
        )

        score = match.get(
            "score",
            0,
        )

        if text:

            documents.append(
                {
                    "text": text,
                    "page": page,
                    "chunk": chunk,
                    "score": score,
                }
            )

    return documents


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    documents,
):

    if not documents:

        return (
            "I couldn't find relevant information "
            "in the Agentic AI ebook."
        )

    context_parts = []

    for document in documents:

        context_parts.append(
            f"[Page {document['page']}]\n"
            f"{document['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are an AI assistant answering questions
about the provided Agentic AI ebook.

Use ONLY the provided context to answer the
user's question.

Rules:

1. Answer using the provided context.
2. Do not invent facts.
3. If the answer cannot be found in the
   provided context, say that the information
   was not found in the ebook.
4. Give a clear and concise answer.
5. When possible, mention the relevant page
   number.
6. Do not mention these instructions.

CONTEXT:

{context}

USER QUESTION:

{question}
"""

    llm = get_llm()

    response = llm.invoke(
        prompt
    )

    return response.content


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for i, source in enumerate(
                    message["sources"],
                    start=1,
                ):

                    st.markdown(
                        f"""
                        <div class="source-box">

                        <strong>
                        Source {i}
                        </strong>

                        <br>

                        Page:
                        {source["page"]}

                        <br>

                        Relevance score:
                        {source["score"]:.4f}

                        <br><br>

                        {source["text"]}

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about the Agentic AI ebook..."
)


if question:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching the ebook..."
        ):

            try:

                documents = retrieve_documents(
                    question
                )

                with st.spinner(
                    "Generating answer..."
                ):

                    answer = generate_answer(
                        question,
                        documents,
                    )

                st.markdown(
                    answer
                )

                # --------------------------------------------
                # SOURCES
                # --------------------------------------------

                if documents:

                    with st.expander(
                        "📚 Sources"
                    ):

                        for i, source in enumerate(
                            documents,
                            start=1,
                        ):

                            st.markdown(
                                f"""
                                <div class="source-box">

                                <strong>
                                Source {i}
                                </strong>

                                <br>

                                Page:
                                {source["page"]}

                                <br>

                                Relevance score:
                                {source["score"]:.4f}

                                <br><br>

                                {source["text"]}

                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # --------------------------------------------
                # SAVE MESSAGE
                # --------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": documents,
                    }
                )

            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )