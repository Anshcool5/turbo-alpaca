from django.shortcuts import render
from django.http import HttpResponse
import gradio as gr
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import threading
from get_keys_from_json import analyze_keys

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the LLM
llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b", groq_api_key=groq_api_key)

def keys_from_json(request):
    master_dict = {
        'Total Sales': [], 'Gross Sales': [], 'Net Sales': [], 'Total Orders': [], 'Discounts': [],
        'Returns': [], 'Shipping': [], 'customer_id': [], 'product_id': [], 'quantity': [],
        'date': [], 'Year': [], 'Month': [], 'cost_price': [], 'stock_level': [], 'expiry_date': []
    }
    # Assuming user.files gives you the File objects
    for f in request.user.files.all():
        file_path = 'turbo/media/uploads/' + f.file_name
        file_dict = analyze_keys(file_path)  # Make sure analyze_keys is imported/defined
        for key, val in file_dict.items():
            if key in master_dict and master_dict[key] == []:
                master_dict[key].append(f.file_name)
                master_dict[key].append(val)
    non_empty_keys = [key for key, value in master_dict.items() if value != []]

def get_llama_response(query: str):
    """Function to process user queries and generate responses."""
    metrics_input = f"""You are a robust and well-trained business advisor for small business owners.
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

    if "</think>" in output:
        after_think = output.split("</think>")[1].strip()
    else:
        after_think = ""

    return after_think

# Create Gradio Interface
def create_gradio_interface():
    """Function to create and launch the Gradio interface."""
    iface = gr.Interface(fn=get_llama_response, inputs="text", outputs="text")
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False, inline=False)

# Start Gradio in a separate thread
threading.Thread(target=create_gradio_interface, daemon=True).start()

def chatbot_view(request):
    """Django View: Embeds the Gradio chatbot using an iframe."""
    # return HttpResponse("working")
    return render(request, "chatbot/chat.html")
