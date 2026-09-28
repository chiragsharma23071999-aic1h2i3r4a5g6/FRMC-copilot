from app.embeddings.embedder import get_embedding_model
from app.vectorstore.chroma_db import get_chroma_client


def get_retriever():

    print("Loading embedding model...")

    embedding_model = get_embedding_model()

    print("Connecting to ChromaDB...")

    client = get_chroma_client()

    collection = client.get_collection(
        name="frmc_controls"
    )

    return embedding_model, collection


def search_documents(query, top_k=3):

    embedding_model, collection = get_retriever()

    print()
    print("Searching for:")
    print(query)

    query_embedding = embedding_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results


if __name__ == "__main__":

    query = input("Enter your question: ")

    results = search_documents(query)

    print()
    print("===================================")
    print("SEARCH RESULTS")
    print("===================================")

    documents = results["documents"][0]

    for i, document in enumerate(documents):

        print()
        print(f"--- Result {i + 1} ---")
        print(document)