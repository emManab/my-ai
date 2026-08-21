import chromadb
from sentence_transformers import SentenceTransformer


DB_PATH = "data/chroma_db"
COLLECTION_NAME = "manab_conversations"
MODEL_NAME = "BAAI/bge-m3"


def main():

    print("=" * 55)
    print("              MANAB AI - RAG TEST")
    print("=" * 55)

    # -------------------------------
    # Load embedding model
    # -------------------------------

    print("\nLoading BGE-M3...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda"
    )

    print("BGE-M3 loaded on GPU.")

    # -------------------------------
    # Connect to ChromaDB
    # -------------------------------

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Database contains "
        f"{collection.count():,} conversations."
    )

    print("\nType 'exit' or 'quit' to stop.")

    # -------------------------------
    # Interactive search
    # -------------------------------

    while True:

        query = input("\nYou: ").strip()

        if query.lower() in {
            "exit",
            "quit"
        }:
            break

        if not query:
            continue

        print("\nSearching your memories...\n")

        # Create embedding for user message
        query_embedding = model.encode(
            query,
            convert_to_tensor=False
        )

        # Search ChromaDB
        results = collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=5,
            include=[
                "documents",
                "distances",
                "metadatas"
            ]
        )

        documents = results["documents"][0]
        distances = results["distances"][0]

        # -------------------------------
        # Display results
        # -------------------------------

        for i, (document, distance) in enumerate(
            zip(documents, distances),
            start=1
        ):

            print(
                f"--- Memory {i} ---"
            )

            print(
                f"Distance: {distance:.4f}"
            )

            print(document)

            print()

        print("-" * 55)

    print("\nRAG test finished.")


if __name__ == "__main__":
    main()