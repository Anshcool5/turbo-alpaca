from chromadb.config import Settings
import chromadb

# Create a new Chroma client with the latest API
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="db"))

# Recreate your collection
collection = client.get_or_create_collection(name="BusinessCollection")

# Add your data manually
collection.add(
    documents=["Document 1", "Document 2", "Document 3"],
    metadatas=[{"source": "Old Database"}, {"source": "Old Database"}, {"source": "Old Database"}],
    ids=["id1", "id2", "id3"]
)

print("Migration completed.")
