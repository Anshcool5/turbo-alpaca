from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from .get_keys_from_json import analyze_keys
from .perform_analysis import determine_and_call_analytics
from .competition_analysis import get_search_query_from_llm, run_playwright
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
    # Revised metrics prompt that includes the actual user query.
    metrics_template = """
    You are a business assistant. Based solely on the user's query, decide whether the user intends to generate plots/reports, requires general business advice or wants to analyze the competition.
    
    If the user's query indicates that they want to generate visual plots, reports, or similar business content, respond with only the word "PLOT".
    Else If the user's query indicates that they want to analyze the competition, respond with only the word "ANALYZE".
    Otherwise, provide an appropriate business advisory response.
    
    User Query: {text}
    
    Your Response: "PLOT", "ANALYZE" or generic business response"""
    
    metrics_prompt = PromptTemplate(
        template=metrics_template,
        input_variables=["text"]
    )
    
    qa1 = LLMChain(llm=llm, prompt=metrics_prompt)
    result1 = qa1.generate([{"text": query}])
    output = result1.generations[0][0].text.strip()
    print(f"Output: {output}")
    
    if output == "PLOT":
        # Iterate through the user's files
        for f in user.files.all():
            if f.file_name not in analyzed_files:
                analyzed_files.append(f.file_name)
            else:
                continue
            file_path = 'media/uploads/' + f.file_name
            file_dict = analyze_keys(file_path)  # Ensure this function is imported/defined
            for key, val in file_dict.items():
                if key in master_dict and not master_dict[key]:
                    master_dict[key].append(f.file_name)
                    master_dict[key].append(val)
        output = determine_and_call_analytics(query, master_dict)     
    elif output == "ANALYZE":
        search_query = get_search_query_from_llm(query)
        run_playwright(search_query)
        # TO-DO convert html to csv
        # Plot csv and give insights?
    return output
