from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import json
from json import JSONDecodeError
import ast
import re

# Load environment variables from .env file
load_dotenv()

# Get the API Key from .env
groq_api_key = os.getenv("GROQ_API_KEY")
open_router_api_key = os.getenv("OPEN_ROUTER_API_KEY")

# Data manipulation
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

# Advanced libraries
from prophet import Prophet
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import apriori, association_rules

# New advanced libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# Load the dataset
data = pd.read_json('data_for_chroma/business.retailsales2.json')
# global Total Sales, Gross Sales, Net Sales, Total Orders, Discounts, Returns, Shipping, customer_id, product_id, quantity, date, Year, Month, cost_price, stock_level, expiry_date


temp = {'Total Sales': None, 'Gross Sales': None, 'Net Sales': None, 'Total Orders': None, 'Discounts': None, 'Returns': None, 'Shipping': None, 
        'customer_id': None, 'product_id': None, 'quantity': None, 'date': None, 'Year': None, 'Month': None, 'cost_price': None, 'stock_level': None, 'expiry_date': None}


# from get_keys_from_json import analyze_keys

# analyzed_keys = analyze_keys()

temp = {'Total Sales': 'Total Sales', 'Gross Sales': 'Gross Sales', 'Net Sales': 'Net Sales', 'Total Orders': 'Total Orders', 'Discounts': 'Discounts', 'Returns': 'Returns', 'Shipping': 'Shipping', 'customer_id': None, 'product_id': None, 'quantity': None, 'date': None, 'Year': 'Year', 'Month': 'Month', 'cost_price': None, 'stock_level': None, 'expiry_date': None}

# for key, value in analyzed_keys.items():
#     if key in temp:
#         temp[key] = value

# print(temp)

# Display the first few rows
data.head()

# imp_key_list = ['Total Sales', 'Gross Sales', 'Net Sales', 'Total Orders', 'Discounts', 'Returns', 'Shipping', 'customer_id', 'product_id', 'quantity', 'date', 'Year', 'Month', 'cost_price', 'stock_level', 'expiry_date']
    # Optionally, show all generated figures if available
    

def calculate_total_revenue_data(sales_df):
    """Calculate total revenue and group by period if possible."""
    if temp["Total Sales"] is None or temp["Total Sales"] not in sales_df.columns:
        return None
    total_revenue = sales_df[temp["Total Sales"]].sum()
    grouped_data = None
    if temp["Year"] in sales_df.columns and temp["Month"] in sales_df.columns:
        df = sales_df.copy()
        df["Period"] = pd.to_datetime(df[temp["Year"]].astype(str) + "-" + 
                                      df[temp["Month"]].astype(str) + "-01")
        grouped_data = df.groupby("Period")[temp["Total Sales"]].sum().reset_index()
    return {"total_revenue": total_revenue, "grouped_data": grouped_data}



def analyze_data(data):
    # Initialize the language model with your API key and parameters.
    llm = ChatGroq(
        temperature=0,
        model_name="deepseek-r1-distill-llama-70b",
        groq_api_key=groq_api_key
    )

    # Define the prompt that instructs the model to provide business insights.
    business_insights_instruction = (
        "You are a robust and well trained business advisor."
        "Based on the provided data, analyze the data and provide valuable business insights. "
        "Highlight key trends, Give short and concise insights"
    )

    # Construct the prompt template.
    prompt_template = (
        "Data: {data}\n\n"
        "{instruction}\n\n"
        "Result = "
    )

    # Create a PromptTemplate with the necessary input variables.
    business_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["data", "instruction"]
    )

    # Generate the response using the LLMChain.
    qa_chain = LLMChain(llm=llm, prompt=business_prompt)
    result = qa_chain.generate([{"data": data, "instruction": business_insights_instruction}])
    output = result.generations[0][0].text

    # print(response_text)

    if "</think>" in output:
        after_think = output.split("</think>")[1].strip()  # Split and take the part after </think>
    else:
        after_think = ""  # Handle the case where </think> is not found

    #print("Text After </think>:")
    return after_think

    
data = calculate_total_revenue_data(data)

print(analyze_data(data))

