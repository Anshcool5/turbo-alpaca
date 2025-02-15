import chromadb
from chromadb.config import Settings

# Create a client
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                    persist_directory="db/"
                                ))

# Create a collection
collection = client.create_collection(name="BusinessCollection", fields=["text"], primary_key="id")

# Add documents to the collection
collection.add(
    documents = ["name of documents",],
    metadatas = [{"source": "student info"},],
    ids = ["id1", "id2", "id3"]
)

# Query the collection
results = collection.query(
    query_texts=["What is the student name?"],
    n_results=2
)

# Print the results
results