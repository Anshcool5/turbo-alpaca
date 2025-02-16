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
data = pd.read_json('business.retailsales.json')

# Display the first few rows
data.head()


def calculate_total_revenue(sales_df):
    return sales_df["Total Sales"].sum()

def calculate_profit_margin(sales_df):
    total_gross = sales_df["Gross Sales"].sum()
    total_net = sales_df["Net Sales"].sum()
    return total_net / total_gross if total_gross > 0 else None

def calculate_number_of_transactions(sales_df):
    return sales_df["Total Orders"].sum()

def calculate_average_sale_value(sales_df):
    num_transactions = calculate_number_of_transactions(sales_df)
    return calculate_total_revenue(sales_df) / num_transactions if num_transactions else 0

def calculate_peak_sales_period(sales_df):
    df = sales_df.copy()
        # Create a datetime using Year and Month (assume day 1)
    df["Period"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
    # Group by Period using Total Sales
    grouped = df.groupby("Period")["Total Sales"].sum()
    peak_day = grouped.idxmax()
    peak_value = grouped.max()
    return peak_day, peak_value

def calculate_seasonal_fluctuations(sales_df):
    df = sales_df.copy()
    df["Period"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
    grouped = df.groupby("Period")["Total Sales"].sum()
    return grouped.std()

def aggregate_customer_feedback(sales_df):
    return sales_df['customer_feedback'].value_counts()

def calculate_customer_churn(sales_df, reference_date=datetime.now()):
    last_purchase = sales_df.groupby('customer_id')['date'].max()
    churned_customers = last_purchase[last_purchase < (reference_date - timedelta(days=30))].count()
    total_customers = sales_df['customer_id'].nunique()
    return churned_customers / total_customers if total_customers else 0

def get_best_sellers(sales_df, top_n=3):
    product_sales = sales_df.groupby('product_id')['quantity'].sum().sort_values(ascending=False)
    return product_sales.head(top_n)

def get_worst_sellers(sales_df, bottom_n=3):
    product_sales = sales_df.groupby('product_id')['quantity'].sum().sort_values()
    return product_sales.head(bottom_n)

def get_stock_levels(inventory_df):
    return inventory_df[['product_id', 'stock_level']]

def forecast_stock(sales_df, inventory_df, days=30):
    df = sales_df.copy()
    df['date_only'] = df['date'].dt.date
    daily_sales = df.groupby(['product_id', 'date_only'])['quantity'].sum().reset_index()
    avg_daily_sales = daily_sales.groupby('product_id')['quantity'].mean()
    forecast = {}
    for product in inventory_df['product_id']:
        current_stock = inventory_df.loc[inventory_df['product_id'] == product, 'stock_level'].values[0]
        expected_sales = avg_daily_sales.get(product, 0) * days
        forecast[product] = current_stock - expected_sales
    return forecast

def suggest_stock_ordering(inventory_df, threshold=50, reorder_amount=100):
    orders = {}
    for _, row in inventory_df.iterrows():
        if row['stock_level'] < threshold:
            orders[row['product_id']] = reorder_amount
    return orders

def calculate_stock_valuation(inventory_df):
    df = inventory_df.copy()
    df['valuation'] = df['stock_level'] * df['cost_price']
    return df[['product_id', 'valuation']]

def check_stock_expiry(inventory_df, days=30, reference_date=datetime.now()):
    df = inventory_df.copy()
    df['days_to_expiry'] = (df['expiry_date'] - reference_date).dt.days
    expiring = df[df['days_to_expiry'] <= days]
    return expiring[['product_id', 'expiry_date', 'days_to_expiry']]

def calculate_stock_spoilage(inventory_df):
    return inventory_df[['product_id', 'spoilage']]

def calculate_stock_returns(sales_df):
    return sales_df["Returns"].sum()




# better at interpreting the seasonal trends than xgboost
def forecast_sales_prophet(sales_df, periods=30):
    # Use aggregated Total Sales over time. Requires Year, Month, and Total Sales.
    if not {"Year", "Month", "Total Sales"}.issubset(sales_df.columns):
        return None, None
    df = sales_df.copy()
    df["Period"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
    grouped = df.groupby("Period")["Total Sales"].sum().reset_index()
    grouped.columns = ["ds", "y"]
    model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=False)
    model.fit(grouped)
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    fig = px.line(forecast, x='ds', y='yhat', title="Prophet Forecast: Total Sales")
    fig.add_scatter(x=grouped['ds'], y=grouped['y'], mode='markers', name='Actual')
    return fig, forecast


def forecast_sales_xgboost(sales_df, periods=30):
    # Use aggregated Total Sales. This is a simple lag-based approach.
    if not {"Year", "Month", "Total Sales"}.issubset(sales_df.columns):
        return None, None, None, None
    df = sales_df.copy()
    df["Period"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
    grouped = df.groupby("Period")["Total Sales"].sum().reset_index()
    grouped = grouped.sort_values("Period").reset_index(drop=True)
    grouped["lag1"] = grouped["Total Sales"].shift(1)
    grouped["lag2"] = grouped["Total Sales"].shift(2)
    grouped["lag3"] = grouped["Total Sales"].shift(3)
    grouped = grouped.dropna().reset_index(drop=True)
    X = grouped[["lag1", "lag2", "lag3"]]
    y = grouped["Total Sales"]
    # Train XGBoost on all data (for demonstration)
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    # Forecast next 'periods' months using the last available lags.
    last_row = grouped.iloc[-1]
    current_lag1 = last_row["Total Sales"]
    current_lag2 = grouped.iloc[-2]["Total Sales"] if len(grouped) >= 2 else current_lag1
    current_lag3 = grouped.iloc[-3]["Total Sales"] if len(grouped) >= 3 else current_lag1
    future_preds = []
    current_date = grouped.iloc[-1]["Period"]
    for _ in range(periods):
        current_date += pd.DateOffset(months=1)
        X_input = np.array([[current_lag1, current_lag2, current_lag3]])
        pred = model.predict(X_input)[0]
        future_preds.append((current_date, pred))
        # Update lags
        current_lag3 = current_lag2
        current_lag2 = current_lag1
        current_lag1 = pred
    future_df = pd.DataFrame(future_preds, columns=["date", "prediction"])
    # Create a simple plot: plot the historical and future predictions.
    fig = px.line(grouped, x="Period", y="Total Sales", title="XGBoost Forecast: Total Sales")
    fig.add_scatter(x=future_df["date"], y=future_df["prediction"], mode="lines", name="Forecasted")
    # Compute error metrics on training data (optional)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    return fig, rmse, r2, future_df

# combining the two forecasts to cover the weaknesses of each model
def combine_forecasts(prophet_df, xgboost_df):
    # Both prophet_df and xgboost_df should have a common datetime column ('ds' for prophet and 'date' for xgboost)
    if prophet_df is None or xgboost_df is None:
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

# current dataset is not suitable for this task
def perform_customer_segmentation(sales_df):
    demographics = get_customer_demographics(sales_df)
    customer_sales = sales_df.groupby('customer_id').agg(
        total_revenue=('transaction_value', 'sum'),
        transaction_count=('transaction_id', 'nunique')
    ).reset_index()
    df = pd.merge(demographics, customer_sales, on='customer_id', how='left').fillna(0)
    df['gender_numeric'] = df['gender'].map({'Male': 0, 'Female': 1})
    features = df[['age', 'total_revenue', 'transaction_count', 'gender_numeric']]
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(features)
    fig = px.scatter(df, x='age', y='total_revenue', color='cluster',
                     hover_data=['transaction_count', 'gender'],
                     title="Customer Segmentation: Age vs Total Revenue")
    return fig, df


def plot_seasonal_decomposition(sales_df):
    # Use the aggregated Total Sales as a time series.
    if not {"Year", "Month", "Total Sales"}.issubset(sales_df.columns):
        return None
    df = sales_df.copy()
    df["Period"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
    grouped = df.groupby("Period")["Total Sales"].sum().asfreq("MS").fillna(0)
    decomposition = seasonal_decompose(grouped, model="additive", period=12)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    decomposition.observed.plot(ax=axes[0], title="Observed")
    decomposition.trend.plot(ax=axes[1], title="Trend")
    decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
    decomposition.resid.plot(ax=axes[3], title="Residual")
    plt.tight_layout()
    return fig


# correlation heatmap to understand relationships between sales metrics
def plot_correlation_heatmap(sales_df):
    # Use the numeric columns that are allowed.
    desired = ["Total Orders", "Gross Sales", "Discounts", "Returns", "Net Sales", "Shipping", "Total Sales"]
    cols = [col for col in desired if col in sales_df.columns]
    if not cols:
        return None
    corr = sales_df[cols].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap: Sales Metrics")
    return fig
