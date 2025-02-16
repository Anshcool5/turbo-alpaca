from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import json
from json import JSONDecodeError
import ast
import time

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

metrics_list = ['total revenue', 'profit margins', 'number of transactions', 'average sale value', 
                'peak sales period', 'seasonal fluctuations', 'customer demographics', 'customer feedback', 
                'customer churn', 'best sellers', 'worst sellers', 'stock levels', 
                 'stock forecasting', 'stock ordering', 'stock tracking', 
                 'stock valuation', 'stock expiry', 'stock spoilage', 'stock returns']

metrics_input = """Based on the given metrics list and the provided dictionary, can you find ALL the metrics I can compute and return them as a list.
                    Go through ALL the metrics in the list."""

metrics_template = """
Human: {text}
Metrics List: {metrics_list}
Dictionary: {dictionary}
Assistant: return the result as a list ONLY
"""

metrics_prompt = PromptTemplate(template=metrics_template,
                        input_variables=["text", "metrics_list", "dictionary"])

qa2 = LLMChain(llm=llm, prompt=metrics_prompt)
result2 = qa2.generate([{"text": metrics_input, "metrics_list": metrics_list, "dictionary": dictionary}])
possible_metrics = ast.literal_eval(result2.generations[0][0].text)
print(possible_metrics)

list_keys = []
for metric in possible_metrics:
    key_input = """Based on the provided dictionary, can you find all the relevant keys to compute the metric {metric} and return them as the 
    value fields in a dictionary with the key being the metric."""

    key_template = """
    Human: {text}
    Dictionary: {dictionary}
    Metric: {metric}
    Assistant: return the result as a dictionary ONLY"""

    key_prompt = PromptTemplate(template=key_template,
                            input_variables=["text", "dictionary", "metric"])

    qa1 = LLMChain(llm=llm, prompt=key_prompt)
    result1 = qa1.generate([{"text": key_input, "dictionary": dictionary, "metric": metric}])
    try:
        list_keys.append(json.loads(result1.generations[0][0].text))
    except JSONDecodeError:
        print(result1.generations[0][0].text)
        pass
    time.sleep(3)

#print(list_keys)