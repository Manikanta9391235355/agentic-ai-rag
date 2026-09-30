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


NAMESPACE = "agentic-ai"


def get_index():

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    return pc.Index(
        PINECONE_INDEX_NAME
    )


def retrieve_context(query):

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    query_vector = embeddings.embed_query(
        query
    )

    index = get_index()

    results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
        namespace=NAMESPACE,
    )

    contexts = []

    for match in results["matches"]:

        metadata = match.get(
            "metadata",
            {}
        )

        text = metadata.get(
            "text",
            ""
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        if text:

            contexts.append(
                f"[Page {page}]\n{text}"
            )

    return "\n\n".join(
        contexts
    )


def ask_llm(question, context):

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
    )

    prompt = f"""
You are an AI assistant answering questions
about the provided Agentic AI ebook.

Use the provided context to answer the question.

Rules:

1. Answer using the provided context.
2. Do not invent information.
3. If the answer is not present in the context,
   clearly say that the information was not found
   in the ebook.
4. Keep the answer clear and useful.
5. Mention relevant page numbers when possible.

CONTEXT:

{context}

QUESTION:

{question}
"""

    response = llm.invoke(
        prompt
    )

    return response.content


def main():

    print("=" * 60)
    print("AGENTIC AI RAG CHATBOT")
    print("=" * 60)

    while True:

        question = input(
            "\nYou: "
        ).strip()

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:

            print(
                "\nGoodbye!"
            )

            break

        print(
            "\nSearching the ebook..."
        )

        context = retrieve_context(
            question
        )

        if not context:

            print(
                "\nNo relevant information found."
            )

            continue

        print(
            "Generating answer..."
        )

        answer = ask_llm(
            question,
            context,
        )

        print(
            "\nAssistant:"
        )

        print(
            answer
        )


if __name__ == "__main__":
    main()