import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


DATASET_FILE = "data/rag_dataset.jsonl"
DB_PATH = "data/chroma_db"
COLLECTION_NAME = "manab_conversations"

MODEL_NAME = "BAAI/bge-m3"

BATCH_SIZE = 256


def load_dataset():

    records = []

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            records.append(json.loads(line))

    return records


def main():

    print("=" * 50)
    print("        MANAB AI - RAG BUILDER")
    print("=" * 50)

    # ----------------------------------------
    # Load embedding model
    # ----------------------------------------

    print("\nLoading BGE-M3...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda"
    )

    print("BGE-M3 loaded on GPU.")

    # ----------------------------------------
    # Load dataset
    # ----------------------------------------

    print("\nLoading dataset...")

    records = load_dataset()

    print(
        f"Loaded {len(records):,} conversations."
    )

    # ----------------------------------------
    # Create ChromaDB
    # ----------------------------------------

    print("\nCreating ChromaDB...")

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Existing documents: {collection.count():,}"
    )

    # ----------------------------------------
    # Process in batches
    # ----------------------------------------

    total = len(records)

    for start in range(0, total, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = records[start:end]

        print(
            f"\nProcessing "
            f"{start + 1:,}-{end:,} "
            f"of {total:,}"
        )

        documents = []
        ids = []
        metadatas = []

        for index, record in enumerate(batch):

            context = record["context"]
            response = record["response"]

            document = (
                f"Person: {context}\n"
                f"Manab: {response}"
            )

            documents.append(document)

            ids.append(
                f"conversation_{start + index}"
            )

            metadatas.append({
                "context": context,
                "response": response,

                "context_message_count": str(
                    record["context_message_count"]
                ),

                "response_message_count": str(
                    record["response_message_count"]
                ),

                "context_has_question": str(
                    record["context_has_question"]
                ),

                "response_has_question": str(
                    record["response_has_question"]
                ),

                "context_has_emoji": str(
                    record["context_has_emoji"]
                ),

                "response_has_emoji": str(
                    record["response_has_emoji"]
                )
            })

        # ------------------------------------
        # Create embeddings
        # ------------------------------------

        embeddings = model.encode(
            documents,
            batch_size=16,
            show_progress_bar=True,
            convert_to_tensor=False
        )

        # ------------------------------------
        # Store in ChromaDB
        # ------------------------------------

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        print(
            f"Stored {end:,}/{total:,}"
        )

    # ----------------------------------------
    # Final result
    # ----------------------------------------

    print("\n" + "=" * 50)
    print("       RAG DATABASE COMPLETE")
    print("=" * 50)

    print(
        f"Total documents: {collection.count():,}"
    )

    print(
        "Database:"
        f" {Path(DB_PATH).resolve()}"
    )


if __name__ == "__main__":
    main()