from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from .data_analysis_func import FUNCTIONS
import pandas as pd

load_dotenv()

# Get the API Key from .env
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b", groq_api_key=groq_api_key)

def determine_and_call_analytics(query: str, master_dict: dict):
    print("sucess!")
    metric_funcs = available_functions_from_metrics(master_dict)
    print("sucess1!")
    metric_funcs_list = metric_funcs.keys()
    metrics_input = f"""Based on the user query: '{query}' and the list of metric functions that can possibly be computed: {metric_funcs_list}, find the metric function needed 
    to address the user's request. If the user's request can be addressed, return the metric function in the same format as the list. Else return NO."""

    metrics_template = """
    Human: {text}
    Assistant: return the metric function in the same format or NO 
    """

    metrics_prompt = PromptTemplate(
        template=metrics_template,
        input_variables=["text"]
    )

    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa1 = LLMChain(llm=llm, prompt=metrics_prompt)
    result1 = qa1.generate([{"text": metrics_input}])
    output = result1.generations[0][0].text
    if output in metric_funcs:
        output = "Sure thing, I'll be generating your plots based on your shared files!"
        data = pd.read_json(metric_funcs[output])
        calc_func_output = FUNCTIONS[output](data)
        output = "\nDone!"

    elif "NO" in output:
        output = "I'm sorry your shared data is either missing key metrics or the  plot is out of my current scope."

    return output

def available_functions_from_metrics(available_metrics):
    """
    Given a list of available metrics (as strings), returns a list of function names
    from the data_analysis_func.py file that can be computed using the available metrics.
    
    The mapping is based on the keys required by each function.
    """
    # Mapping of function names to the required metrics
    functions_requirements = {
        "calculate_total_revenue_data": {"Total Sales"},
        "calculate_profit_margin_data": {"Gross Sales", "Net Sales"},
        "calculate_number_of_transactions_data": {"Total Orders"},
        "calculate_peak_sales_period_data": {"Year", "Month", "Total Sales"},
        "calculate_seasonal_fluctuations_data": {"Year", "Month", "Total Sales"},
        "calculate_customer_churn_data": {"customer_id", "date"},
        "get_best_sellers_data": {"product_id", "quantity"},
        "get_worst_sellers_data": {"product_id", "quantity"},
        "get_stock_levels_data": {"product_id", "stock_level"},
        "forecast_stock_data": {"date", "product_id", "quantity", "stock_level"},
        "suggest_stock_ordering_data": {"stock_level", "product_id"},
        "calculate_stock_valuation_data": {"stock_level", "cost_price", "product_id"},
        "check_stock_expiry_data": {"expiry_date", "product_id"},
        "calculate_stock_returns_data": {"Returns"},
        "forecast_sales_prophet_data": {"Year", "Month", "Total Sales"},
        "get_customer_demographics": {"customer_id"},
        "perform_customer_segmentation_data": {"customer_id"},
        "seasonal_decomposition_data": {"Year", "Month", "Total Sales"}
    }
    
    available = {}
    for func_name, req_metrics in functions_requirements.items():
        # Check if every required metric is present in the available_metrics list
        if all(metric in available_metrics.keys() for metric in req_metrics):
            available[func_name] = [available_metrics[metric][0] for metric in req_metrics if metric in available_metrics]
    return available