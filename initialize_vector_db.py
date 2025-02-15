from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
#from langchain.chains import RetrievalQA
#from langchain.prompts import PromptTemplate
#from llama_parse import LlamaParse
#from decouple import Config, RepositoryEnv
from langchain_community.document_loaders import TextLoader
import chromadb
from chromadb import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

ef = HuggingFaceEmbeddings()

class DefChromaEF(Embeddings):
  def __init__(self,ef):
    self.ef = ef

  def embed_documents(self,doc):
    return self.ef.embed_documents(doc)

  def embed_query(self, query):
    return self.ef.embed_query([query])[0]
  
  def __call__(self, input):
    return self.embed_documents(input)

embedding_function=DefChromaEF(ef)

document_path = "/Users/anshulverma/Documents/AI_ML_LLMs/Course Planner/course_info_SC_CMPUT.txt"
loader = TextLoader(document_path)
loaded_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=6144, chunk_overlap=128)
documents = text_splitter.split_documents(loaded_documents)
print(len(documents))

client = chromadb.Client()
client.delete_collection(name="CMPUT_Courses")
collection = client.get_or_create_collection(name="CMPUT_Courses", embedding_function=embedding_function)
docs_json = []
for doc in documents:
    doc_json = doc.page_content
    docs_json.append(doc_json)
id_list = list(map(lambda tup: f"id{tup[0]}", enumerate(docs_json)))
collection.add(
    documents = docs_json,
    ids = id_list)

chroma = Chroma(client=client, collection_name="CMPUT_Courses", embedding_function=ef)
retriever = chroma.as_retriever(k=5, search_type="mmr")