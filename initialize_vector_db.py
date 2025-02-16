from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
import chromadb
from chromadb.api import Embeddings
from pprint import pprint

# Initialize HuggingFaceEmbeddings with a specific model
ef = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Explicitly specify the model name

# Define custom embedding function
class DefChromaEF(Embeddings):
    def __init__(self, ef):
        self.ef = ef

    def embed_documents(self, doc):
        return self.ef.embed_documents(doc)

    def embed_query(self, query):
        return self.ef.embed_query([query])[0]

    def __call__(self, input):
        return self.embed_documents(input)

embedding_function = DefChromaEF(ef)

# Load the documents
document_path = "test.txt"
loader = TextLoader(document_path)
loaded_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=6144, chunk_overlap=128)
documents = text_splitter.split_documents(loaded_documents)
print(f"Number of documents: {len(documents)}")

loader = JSONLoader(
    file_path='data_for_chroma/business.retailsales.json',
    jq_schema=".",  # Load the entire JSON structure
    text_content=False
)
data = loader.load()
#pprint(data)

# Initialize Chroma client
client = chromadb.Client()

print(f"Loaded {len(data)} documents from JSON.")

# Step 6: Check if the Collection Exists Before Deleting
collections = client.list_collections()
if "RetailSales" in [col.name for col in collections]:
    print("Deleting existing collection...")
    client.delete_collection(name="RetailSales")
else:
    print("Collection 'RetailSales' does not exist. Skipping deletion.")

# Step 7: Create or Get the Collection
collection = client.get_or_create_collection(name="RetailSales", embedding_function=embedding_function)

# Add documents to the collection
#docs_json = [doc.page_content for doc in documents]
docs_json = [doc.page_content for doc in data]
id_list = [f"id{i}" for i, _ in enumerate(docs_json)]
collection.add(
    documents=docs_json,
    ids=id_list
)

# Initialize Chroma
chroma = Chroma(client=client, collection_name="RetailSales", embedding_function=ef)

# Initialize retriever
retriever = chroma.as_retriever(k=5, search_type="mmr")

# Test query
query = "What are the sales trends for retail businesses?"
results = retriever.get_relevant_documents(query)

# Step 12: Display Query Results
print("Query Results:")
for result in results:
    print(result.page_content)