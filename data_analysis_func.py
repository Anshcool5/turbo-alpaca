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

def calculate_total_revenue(sales_df):
    if temp["Total Sales"] is None or temp["Total Sales"] not in sales_df.columns:
        return None
    return sales_df[temp["Total Sales"]].sum()

def calculate_profit_margin(sales_df):
    if (temp["Gross Sales"] is None or temp["Gross Sales"] not in sales_df.columns or
        temp["Net Sales"] is None or temp["Net Sales"] not in sales_df.columns):
        return None
    total_gross = sales_df[temp["Gross Sales"]].sum()
    total_net = sales_df[temp["Net Sales"]].sum()
    return total_net / total_gross if total_gross > 0 else None

def calculate_number_of_transactions(sales_df):
    if temp["Total Orders"] is None or temp["Total Orders"] not in sales_df.columns:
        return None
    return sales_df[temp["Total Orders"]].sum()

def calculate_average_sale_value(sales_df):
    # Relying on the above functions to check required columns.
    num_transactions = calculate_number_of_transactions(sales_df)
    total_revenue = calculate_total_revenue(sales_df)
    if num_transactions is None or total_revenue is None or num_transactions == 0:
        return None
    return total_revenue / num_transactions

def calculate_peak_sales_period(sales_df):
    if (temp["Year"] is None or temp["Year"] not in sales_df.columns or
        temp["Month"] is None or temp["Month"] not in sales_df.columns or
        temp["Total Sales"] is None or temp["Total Sales"] not in sales_df.columns):
        return None
    df = sales_df.copy()
    # Create a datetime using Year and Month (assume day 1)
    df["Period"] = pd.to_datetime(df[temp["Year"]].astype(str) + "-" + df[temp["Month"]].astype(str) + "-01")
    # Group by Period using Total Sales
    grouped = df.groupby("Period")[temp["Total Sales"]].sum()
    peak_day = grouped.idxmax()
    peak_value = grouped.max()
    return peak_day, peak_value


def calculate_seasonal_fluctuations(sales_df):
    if (temp["Year"] is None or temp["Year"] not in sales_df.columns or
        temp["Month"] is None or temp["Month"] not in sales_df.columns or
        temp["Total Sales"] is None or temp["Total Sales"] not in sales_df.columns):
        return None
    df = sales_df.copy()
    df["Period"] = pd.to_datetime(df[temp["Year"]].astype(str) + "-" + df[temp["Month"]].astype(str) + "-01")
    grouped = df.groupby("Period")[temp["Total Sales"]].sum()
    return grouped.std()


def calculate_customer_churn(sales_df, reference_date=datetime.now()):
    if (temp["customer_id"] is None or temp["customer_id"] not in sales_df.columns or
        temp["date"] is None or temp["date"] not in sales_df.columns):
        return None
    last_purchase = sales_df.groupby(temp["customer_id"])[temp["date"]].max()
    churned_customers = last_purchase[last_purchase < (reference_date - timedelta(days=30))].count()
    total_customers = sales_df[temp["customer_id"]].nunique()
    return churned_customers / total_customers if total_customers else None

def get_best_sellers(sales_df, top_n=3):
    if (temp["product_id"] is None or temp["product_id"] not in sales_df.columns or
        temp["quantity"] is None or temp["quantity"] not in sales_df.columns):
        return None
    product_sales = sales_df.groupby(temp["product_id"])[temp["quantity"]].sum().sort_values(ascending=False)
    return product_sales.head(top_n)

def get_worst_sellers(sales_df, bottom_n=3):
    if (temp["product_id"] is None or temp["product_id"] not in sales_df.columns or
        temp["quantity"] is None or temp["quantity"] not in sales_df.columns):
        return None
    product_sales = sales_df.groupby(temp["product_id"])[temp["quantity"]].sum().sort_values()
    return product_sales.head(bottom_n)

def get_stock_levels(inventory_df):
    if (temp["product_id"] is None or temp["product_id"] not in inventory_df.columns or
        temp["stock_level"] is None or temp["stock_level"] not in inventory_df.columns):
        return None
    return inventory_df[[temp["product_id"], temp["stock_level"]]]

def forecast_stock(sales_df, inventory_df, days=30):
    if (temp["date"] is None or temp["date"] not in sales_df.columns or
        temp["product_id"] is None or temp["product_id"] not in sales_df.columns or
        temp["quantity"] is None or temp["quantity"] not in sales_df.columns):
        return None
    if (temp["product_id"] is None or temp["product_id"] not in inventory_df.columns or
        temp["stock_level"] is None or temp["stock_level"] not in inventory_df.columns):
        return None
    df = sales_df.copy()
    df['date_only'] = df[temp["date"]].dt.date
    daily_sales = df.groupby([temp["product_id"], 'date_only'])[temp["quantity"]].sum().reset_index()
    avg_daily_sales = daily_sales.groupby(temp["product_id"])[temp["quantity"]].mean()
    forecast = {}
    for product in inventory_df[temp["product_id"]]:
        current_stock = inventory_df.loc[inventory_df[temp["product_id"]] == product, temp["stock_level"]].values[0]
        expected_sales = avg_daily_sales.get(product, 0) * days
        forecast[product] = current_stock - expected_sales
    return forecast

def suggest_stock_ordering(inventory_df, threshold=50, reorder_amount=100):
    if (temp["stock_level"] is None or temp["stock_level"] not in inventory_df.columns or
        temp["product_id"] is None or temp["product_id"] not in inventory_df.columns):
        return None
    orders = {}
    for _, row in inventory_df.iterrows():
        if row[temp["stock_level"]] < threshold:
            orders[row[temp["product_id"]]] = reorder_amount
    return orders

def calculate_stock_valuation(inventory_df):
    if (temp["stock_level"] is None or temp["stock_level"] not in inventory_df.columns or
        temp["cost_price"] is None or temp["cost_price"] not in inventory_df.columns or
        temp["product_id"] is None or temp["product_id"] not in inventory_df.columns):
        return None
    df = inventory_df.copy()
    df['valuation'] = df[temp["stock_level"]] * df[temp["cost_price"]]
    return df[[temp["product_id"], 'valuation']]

def check_stock_expiry(inventory_df, days=30, reference_date=datetime.now()):
    if (temp["expiry_date"] is None or temp["expiry_date"] not in inventory_df.columns or
        temp["product_id"] is None or temp["product_id"] not in inventory_df.columns):
        return None
    df = inventory_df.copy()
    df['days_to_expiry'] = (df[temp["expiry_date"]] - reference_date).dt.days
    expiring = df[df['days_to_expiry'] <= days]
    return expiring[[temp["product_id"], temp["expiry_date"], 'days_to_expiry']]

def calculate_stock_spoilage(inventory_df):
    if (temp["product_id"] is None or temp["product_id"] not in inventory_df.columns or
        temp["spoilage"] is None or temp["spoilage"] not in inventory_df.columns):
        return None
    return inventory_df[[temp["product_id"], temp["spoilage"]]]

def calculate_stock_returns(sales_df):
    if temp["Returns"] is None or temp["Returns"] not in sales_df.columns:
        return None
    return sales_df[temp["Returns"]].sum()


def forecast_sales_prophet(sales_df, periods=30):
    if (temp["Year"] is None or temp["Year"] not in sales_df.columns or
        temp["Month"] is None or temp["Month"] not in sales_df.columns or
        temp["Total Sales"] is None or temp["Total Sales"] not in sales_df.columns):
        return None, None
    df = sales_df.copy()
    df["Period"] = pd.to_datetime(df[temp["Year"]].astype(str) + "-" + df[temp["Month"]].astype(str) + "-01")
    grouped = df.groupby("Period")[temp["Total Sales"]].sum().reset_index()
    grouped.columns = ["ds", "y"]
    model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=False)
    model.fit(grouped)
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    fig = px.line(forecast, x='ds', y='yhat', title="Prophet Forecast: Total Sales")
    fig.add_scatter(x=grouped['ds'], y=grouped['y'], mode='markers', name='Actual')
    return fig, forecast

def forecast_sales_xgboost(sales_df, periods=3):
    if (temp["Year"] is None or temp["Year"] not in sales_df.columns or
        temp["Month"] is None or temp["Month"] not in sales_df.columns or
        temp["Total Sales"] is None or temp["Total Sales"] not in sales_df.columns):
        return None, None, None, None

    # print("dvssbsvajfhvakhckivaikhas")

    df = sales_df.copy()
    df["Period"] = pd.to_datetime(df[temp["Year"]].astype(str) + "-" + df[temp["Month"]].astype(str) + "-01")
    grouped = df.groupby("Period")[temp["Total Sales"]].sum().reset_index().sort_values("Period")
    
    # Create lag features
    for i in range(1, 4):
        grouped[f"lag{i}"] = grouped[temp["Total Sales"]].shift(i)
    grouped = grouped.dropna().reset_index(drop=True)
    
    X = grouped[["lag1", "lag2", "lag3"]]
    y = grouped[temp["Total Sales"]]
    
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Forecast future periods using the last available lags
    current_lags = [
        grouped.iloc[-1][temp["Total Sales"]],
        grouped.iloc[-2][temp["Total Sales"]] if len(grouped) >= 2 else grouped.iloc[-1][temp["Total Sales"]],
        grouped.iloc[-3][temp["Total Sales"]] if len(grouped) >= 3 else grouped.iloc[-1][temp["Total Sales"]]
    ]
    
    future_preds = []
    current_date = grouped.iloc[-1]["Period"]
    for _ in range(periods):
        current_date += pd.DateOffset(months=1)
        X_input = np.array([current_lags])
        pred = model.predict(X_input)[0]
        future_preds.append((current_date, pred))
        current_lags = [pred, current_lags[0], current_lags[1]]
    
    future_df = pd.DataFrame(future_preds, columns=["date", "prediction"])
    
    fig = px.line(grouped, x="Period", y=temp["Total Sales"], title="XGBoost Forecast: Total Sales")
    fig.add_scatter(x=future_df["date"], y=future_df["prediction"], mode="lines", name="Forecasted")
    
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    return fig, rmse, r2, future_df

def combine_forecasts(prophet_df, xgboost_df):
    # Check that both forecast DataFrames have the required columns.
    if (prophet_df is None or xgboost_df is None or 
        "ds" not in prophet_df.columns or "yhat" not in prophet_df.columns or 
        "date" not in xgboost_df.columns or "prediction" not in xgboost_df.columns):
        return None, None
    p_df = prophet_df[["ds", "yhat"]].copy()
    x_df = xgboost_df[["date", "prediction"]].copy()
    x_df.columns = ["ds", "xgb_pred"]
    merged = pd.merge(p_df, x_df, on="ds", how="inner")
    merged["combined"] = (merged["yhat"] + merged["xgb_pred"]) / 2
    fig = px.line(merged, x="ds", y="combined", title="Ensemble Forecast (Prophet + XGBoost)")
    fig.add_scatter(x=merged["ds"], y=merged["yhat"], mode="lines", name="Prophet")
    fig.add_scatter(x=merged["ds"], y=merged["xgb_pred"], mode="lines", name="XGBoost")
    return fig, merged

def perform_customer_segmentation(sales_df):
    # Assume get_customer_demographics is defined elsewhere.
    if (temp["customer_id"] is None or temp["customer_id"] not in sales_df.columns or
        temp["transaction_value"] is None or temp["transaction_value"] not in sales_df.columns or
        temp["transaction_id"] is None or temp["transaction_id"] not in sales_df.columns):
        return None, None
    demographics = get_customer_demographics(sales_df)
    customer_sales = sales_df.groupby(temp["customer_id"]).agg(
        total_revenue=(temp["transaction_value"], 'sum'),
        transaction_count=(temp["transaction_id"], 'nunique')
    ).reset_index()
    df = pd.merge(demographics, customer_sales, on=temp["customer_id"], how='left').fillna(0)
    df['gender_numeric'] = df['gender'].map({'Male': 0, 'Female': 1})
    features = df[['age', 'total_revenue', 'transaction_count', 'gender_numeric']]
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(features)
    fig = px.scatter(df, x='age', y='total_revenue', color='cluster',
                     hover_data=['transaction_count', 'gender'],
                     title="Customer Segmentation: Age vs Total Revenue")
    return fig, df

def plot_seasonal_decomposition(sales_df):
    if (temp["Year"] is None or temp["Year"] not in sales_df.columns or
        temp["Month"] is None or temp["Month"] not in sales_df.columns or
        temp["Total Sales"] is None or temp["Total Sales"] not in sales_df.columns):
        return None
    df = sales_df.copy()
    df["Period"] = pd.to_datetime(df[temp["Year"]].astype(str) + "-" + df[temp["Month"]].astype(str) + "-01")
    grouped = df.groupby("Period")[temp["Total Sales"]].sum().asfreq("MS").fillna(0)
    decomposition = seasonal_decompose(grouped, model="additive", period=12)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    decomposition.observed.plot(ax=axes[0], title="Observed")
    decomposition.trend.plot(ax=axes[1], title="Trend")
    decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
    decomposition.resid.plot(ax=axes[3], title="Residual")
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(sales_df):
    numeric_cols = sales_df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.any():
        return None
    corr = sales_df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap: Sales Metrics")
    return fig


# Run forecast using XGBoost model
# fig_xgb, rmse_xgb, r2_xgb, forecast_xgb_df = forecast_sales_xgboost(data, periods=30)

# # Run forecast using Prophet model
# fig_prophet, forecast_prophet_df = forecast_sales_prophet(data, periods=30)

# # Combine the Prophet and XGBoost forecasts
# fig_combined, combined_df = combine_forecasts(forecast_prophet_df, forecast_xgb_df)

# # Display the combined forecast figure
# fig_combined.show()




# use output to store the results of the analysis and feed the dictionary to the chroma API
# output = {
#     "Total Revenue": calculate_total_revenue(data),
#     "Profit Margin": calculate_profit_margin(data),
#     "Number of Transactions": calculate_number_of_transactions(data),
#     "Average Sale Value": calculate_average_sale_value(data),
#     "Peak Sales Period": calculate_peak_sales_period(data),
#     "Seasonal Fluctuations": calculate_seasonal_fluctuations(data),
#     "XGBoost RMSE": rmse_xgb,
#     "XGBoost R2": r2_xgb,
#     "Forecast XGBoost Data": forecast_xgb_df.to_dict(orient="records"),
#     "Forecast Prophet Data": forecast_prophet_df.to_dict(orient="records"),
#     "Combined Forecast Data": combined_df.to_dict(orient="records")
# }
