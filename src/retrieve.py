from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)


NAMESPACE = "agentic-ai"


def get_index():

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    index = pc.Index(
        PINECONE_INDEX_NAME
    )

    return index


def search_documents(query):

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

    return results


def main():

    query = input(
        "\nEnter your question: "
    )

    results = search_documents(
        query
    )

    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS")
    print("=" * 60)

    for i, match in enumerate(
        results["matches"],
        start=1,
    ):

        metadata = match.get(
            "metadata",
            {}
        )

        print()
        print(
            f"Result {i}"
        )

        print(
            f"Score: {match['score']}"
        )

        print(
            f"Page: "
            f"{metadata.get('page')}"
        )

        print(
            f"Chunk: "
            f"{metadata.get('chunk')}"
        )

        print(
            f"\n{metadata.get('text')}"
        )


if __name__ == "__main__":
    main()