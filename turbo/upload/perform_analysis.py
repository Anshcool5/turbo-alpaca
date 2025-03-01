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
    metrics_input = f"""Based on the user query: '{query}' and the list of metric functions: {metric_funcs_list}, determine which metric function should be used. 
    If the user's request can be addressed, ONLY return the function name exactly as it appears in the list. If it cannot be addressed, return NO.
    Please enclose any reasoning (chain-of-thought) within <think> and </think> tags, and then on a new line after </think> output ONLY the function name or NO.
    """

    metrics_template = """
    Human: {text}
    Assistant: ONLY return the function name exactly as it appears in the list.
    """

    # Create a prompt template that expects the final answer on the line after </think>
    metrics_prompt = PromptTemplate(
        template=metrics_template,
        input_variables=["text"]
    )

    qa1 = LLMChain(llm=llm, prompt=metrics_prompt)
    result1 = qa1.generate([{"text": metrics_input}])
    output = result1.generations[0][0].text.strip()

    # If the output includes chain-of-thought tags, extract the part after </think>
    if "</think>" in output:
        final_output = output.split("</think>")[-1].strip()
    else:
        final_output = output.strip()

    print(f"Final output: {final_output}")
    final_output = extract_final_function_name(final_output, metric_funcs_list)
    
    print("final_output", final_output)
    if final_output in metric_funcs:
        print("I am gonna cum")
        # proceed with using final_output as the function name

        output = "Sure thing, I'll be generating your plots based on your shared files!"
        data = pd.read_json('media/uploads/' + metric_funcs[final_output][0])
        calc_func_output = FUNCTIONS[final_output](data)
        print(calc_func_output)
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
        # Ensure every required metric is present
        if all(metric in available_metrics for metric in req_metrics):
            values = []
            for metric in req_metrics:
                metric_value = available_metrics[metric]
                # If the value is a list and not empty, use its first element.
                if isinstance(metric_value, list):
                    if metric_value:  # non-empty list
                        values.append(metric_value[0])
                    else:
                        # Handle empty list scenario (e.g. append None or skip)
                        continue
                else:
                    # Otherwise, assume it's a direct value.
                    values.append(metric_value)
            available[func_name] = values
    return available

def extract_final_function_name(llm_output: str, metric_funcs: list) -> str:
    """
    Extracts the final non-empty line from the LLM output and returns it
    if it matches one of the known metric functions.
    """
    # Split the output into lines and iterate backwards.
    lines = [line.strip() for line in llm_output.strip().splitlines() if line.strip()]
    for line in reversed(lines):
        if line in metric_funcs:
            return line
    return None