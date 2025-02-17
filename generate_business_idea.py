from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load environment variables from .env file
load_dotenv()

# Get the API Key from .env
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b", groq_api_key=groq_api_key)

qa2 = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type="stuff")
result2 = qa2({"query": question})
print(result2['result'])