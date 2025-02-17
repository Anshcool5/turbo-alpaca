from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

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

def get_llama_response(query: str):

    metrics_input = f"""You are a robust and well trained business advisor for small business owners.
                    Ask the user for relevant documents if needed to aid your advising based on the query: '{query}'."""

    metrics_template = """
    Chat History: {chat_history}
    Human: {text}
    Assistant: ONLY return the relevant documents needed to aid your analysis, else return an appropriate response. 
    """

    metrics_prompt = PromptTemplate(
        template=metrics_template,
        input_variables=["chat_history", "text"]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa1 = LLMChain(llm=llm, prompt=metrics_prompt, memory=memory)
    result1 = qa1.generate([{"text": metrics_input, "chat_history": ""}])
    output = result1.generations[0][0].text
    #print(output)

    if "</think>" in output:
        after_think = output.split("</think>")[1].strip()  # Split and take the part after </think>
    else:
        after_think = ""  # Handle the case where </think> is not found

    #print("Text After </think>:")
    return after_think



import gradio as gr
#from langchain_core.runnables import Runnable
#runnable: Runnable = prompt | llm cd

test = gr.Interface(fn=get_llama_response, inputs="text", outputs="text")
test.launch()

#to-do:
#1. Store requested relevant documents
#2. Check the first dictionary from the JSON and see which category it fits
#3. check if all the requested documents are provided, if yes, perform analysis
#4. Else, perform possible analysis and inform user.
#5. every csv might have a distnct column structure, how to perform the necessary forecast on it?