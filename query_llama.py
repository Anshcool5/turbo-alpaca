from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get the API Key from .env
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

# Input text
#input_text = "I want to analyse the curent sales  of my business"
# Query the LLM
#response = llm.invoke(input_text)
# Print the response
#print(response.content)

query = "I want you to forecast my sales"

metrics_input = f"""You are a robust and well trained business advisor for business owners.
                Ask the user for relevant documents to aid your advising based on the query: '{query}'."""

metrics_template = """
Human: {text}
Assistant: ask the users for relevant documents to aid your analysis.
"""

metrics_prompt = PromptTemplate(template=metrics_template,
                        input_variables=["text"])

qa1 = LLMChain(llm=llm, prompt=metrics_prompt)
result1 = qa1.generate([{"text": metrics_input}])
print(result1.generations[0][0].text)