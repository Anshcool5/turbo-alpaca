from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import re
import pprint

idea_name = 'Hi Tea'
business_idea = "Bubble tea shop with taiwanese fried chicken"
relevant_industry = "Food and Beverage"



# Load environment variables from .env file
load_dotenv()

# Get the API Key from .env
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b", groq_api_key=groq_api_key)

# Input text
#input_text = "I want to analyse the curent sales  of my business"
# Query the LLM
#response = llm.invoke(input_text)
# Print the response
#print(response.content)


metrics_input = f"""You are a robust and well trained business advisor for business owners.
                Evaluate the Idea based on Risk, competitiveness, setup cost, expected ROI, scalability and return the results based on the business Idea: '{business_idea}'
                and the releavnt industry: {relevant_industry}."""


metrics_template = """
Human: {text}
Assistant: = Return the relevant score on a scale of 1-10 for each of the 5 metrics and your reasoning as a robust and well trained business advisor.
"""

metrics_prompt = PromptTemplate(template=metrics_template,
                        input_variables=["text"])

qa1 = LLMChain(llm=llm, prompt=metrics_prompt)
result1 = qa1.generate([{"text": metrics_input}])
output = result1.generations[0][0].text
# print(output)

if "</think>" in output:
    after_think = output.split("</think>")[1].strip()  # Split and take the part after </think>
else:
    after_think = output.strip()

numbers_before_slash = re.findall(r"(\d+)\s*/\s*\d+", after_think)

pattern = r"\d+\.\s*\*\*(.*?)\*\*"
matches = re.findall(pattern, after_think)

metric_values = {key: value for key, value in zip(matches, numbers_before_slash)}

print(metric_values)