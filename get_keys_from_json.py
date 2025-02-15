from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import json

# Load environment variables from .env file
load_dotenv()

# Get the API Key from .env
groq_api_key = os.getenv("GROQ_API_KEY")
open_router_api_key = os.getenv("OPEN_ROUTER_API_KEY")

# Specify the path to the JSON file
file_path = 'data_for_chroma/business.retailsales.json'

# Open and load the JSON file
with open(file_path, mode='r', encoding='utf-8') as json_file:
    data = json.load(json_file)  # Load the JSON data as a Python list/dict

# If you want to access the first dictionary from the JSON list:
dictionary = data[0]

llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

prompt_input = """Based on the provided dictionary, can you find the relevant key for the sales figure and return it."""

classify_template = """
Human: {text}
Dictionary: {dictionary}
Assistant: return ONLY the relevant key"""

classify_prompt = PromptTemplate(template=classify_template,
                        input_variables=["text", "dictionary"])

qa1 = LLMChain(llm=llm, prompt=classify_prompt)
result1 = qa1.generate([{"text": prompt_input, "dictionary": dictionary}])
print(result1.generations[0][0].text)