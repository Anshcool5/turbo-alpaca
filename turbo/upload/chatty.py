from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from ...get_keys_from_json import analyze_keys
from ...perform_analysis import determine_and_call_analytics
#from ...explain_plot import 

load_dotenv()

# Get the API Key from .env
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

master_dict = {
        'Total Sales': [], 'Gross Sales': [], 'Net Sales': [], 'Total Orders': [], 'Discounts': [],
        'Returns': [], 'Shipping': [], 'customer_id': [], 'product_id': [], 'quantity': [],
        'date': [], 'Year': [], 'Month': [], 'cost_price': [], 'stock_level': [], 'expiry_date': []
    }

analyzed_files = []

def run_llm(query: str, user):
    metrics_input = f"""You are a robust and well trained business advisor for business owners.
                    Analyze the user query: '{query}'. If the query is asking you to generate/create content, return
                    the word PLOT only. If the query is asking you to analyse/explain exisiting plots or plots in general, return the word
                    EXPLAIN only. Else respond with an appropriate response based on your business advising expertise."""

    metrics_template = """
    Human: {text}
    Assistant: return the word PLOT, EXPLAIN or a generic response based on your business advising expertise. 
    """

    metrics_prompt = PromptTemplate(
        template=metrics_template,
        input_variables=["text"]
    )

    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa1 = LLMChain(llm=llm, prompt=metrics_prompt)
    result1 = qa1.generate([{"text": metrics_input}])
    output = result1.generations[0][0].text
    
    if output == "PLOT":
        # Iterate through the user's files
        for f in user.files.all():
            if f.file_name not in analyzed_files:
                analyzed_files.append(f.file_name)
            else:
                continue
            file_path = 'turbo/media/uploads/' + f.file_name
            file_dict = analyze_keys(file_path)  # Ensure this function is imported/defined
            for key, val in file_dict.items():
                if key in master_dict and not master_dict[key]:
                    master_dict[key].append(f.file_name)
                    master_dict[key].append(val)
        key_list = [key for key, value in master_dict.items() if value != []]
        output = determine_and_call_analytics(query, key_list)

    #elif output == "EXPLAIN":
    return output
