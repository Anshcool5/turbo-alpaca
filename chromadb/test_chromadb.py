import chromadb

# Create a new Chroma client with the latest API
chroma_client = chromadb.Client()

# Recreate your collection
collection = chroma_client.create_collection(name="my_collection")

# Add your data manually
collection.add(
    documents=["Document 1", "Document 2", "Document 3"],
    metadatas=[{"source": "Old Database"}, {"source": "Old Database"}, {"source": "Old Database"}],
    ids=["id1", "id2", "id3"]
)

print("Migration completed.")
