from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.api import Embeddings

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
document_path = "/Users/anshulverma/Documents/AI_ML_LLMs/Course Planner/course_info_SC_CMPUT.txt"
loader = TextLoader(document_path)
loaded_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=6144, chunk_overlap=128)
documents = text_splitter.split_documents(loaded_documents)
print(f"Number of documents: {len(documents)}")

# Initialize Chroma client
client = chromadb.Client()

# Check if the collection exists before deleting
collections = client.list_collections()
if "CMPUT_Courses" in [col.name for col in collections]:
    print("Deleting existing collection...")
    client.delete_collection(name="CMPUT_Courses")
else:
    print("Collection CMPUT_Courses does not exist. Skipping deletion.")

# Create or get the collection
collection = client.get_or_create_collection(name="CMPUT_Courses", embedding_function=embedding_function)

# Add documents to the collection
docs_json = [doc.page_content for doc in documents]
id_list = [f"id{i}" for i, _ in enumerate(docs_json)]
collection.add(
    documents=docs_json,
    ids=id_list
)

# Initialize Chroma
chroma = Chroma(client=client, collection_name="CMPUT_Courses", embedding_function=ef)

# Initialize retriever
retriever = chroma.as_retriever(k=5, search_type="mmr")

# Test query
query = "What is CMPUT 101?"  # Replace with a relevant query
results = retriever.get_relevant_documents(query)
print("Query Results:")
for result in results:
    print(result.page_content)
