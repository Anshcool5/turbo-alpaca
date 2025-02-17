# Data manipulation
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

# Advanced libraries
from prophet import Prophet
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose

# New advanced libraries
import nltk

import os
from django.conf import settings


# Load the dataset
# data = pd.read_json('data_for_chroma/business.retailsales2.json')
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
# data.head()

# imp_key_list = ['Total Sales', 'Gross Sales', 'Net Sales', 'Total Orders', 'Discounts', 'Returns', 'Shipping', 'customer_id', 'product_id', 'quantity', 'date', 'Year', 'Month', 'cost_price', 'stock_level', 'expiry_date']
    # Optionally, show all generated figures if available
    


# =========================
# SALES FUNCTIONS
# =========================

def save_plotly_figure(fig, plot_name_prefix):
    """
    Save the given Plotly figure as a standalone HTML file in the media/plots folder.
    Returns the URL to the saved file.
    """
    # Define the directory for plots inside MEDIA_ROOT
    plot_dir = os.path.join(settings.MEDIA_ROOT, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create a unique filename using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{plot_name_prefix}_{timestamp}.html"
    file_path = os.path.join(plot_dir, file_name)
    
    # Save the figure as a standalone HTML file
    fig.write_html(file_path)
    
    # Build and return the URL where the file can be accessed.
    file_url = os.path.join(settings.MEDIA_URL, "plots", file_name)
    return file_url


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

def plot_total_revenue(grouped_data):
    """Plot total revenue over time and save as HTML."""
    if grouped_data is None:
        return None
    fig = px.bar(grouped_data, x="Period", y=temp["Total Sales"],
                 title="Total Revenue Over Time")
    # Save the figure and return the URL
    return save_plotly_figure(fig, "total_revenue")


def calculate_profit_margin_data(sales_df):
    """Compute gross and net sales and the profit margin."""
    if (temp["Gross Sales"] is None or temp["Gross Sales"] not in sales_df.columns or
        temp["Net Sales"] is None or temp["Net Sales"] not in sales_df.columns):
        return None
    total_gross = sales_df[temp["Gross Sales"]].sum()
    total_net = sales_df[temp["Net Sales"]].sum()
    profit_margin = total_net / total_gross if total_gross > 0 else None
    return {"total_gross": total_gross, "total_net": total_net, "profit_margin": profit_margin}

def plot_profit_margin(data_dict):
    """Plot a bar chart comparing gross vs net sales and save as HTML."""
    if data_dict is None:
        return None
    fig = px.bar(x=["Gross Sales", "Net Sales"],
                 y=[data_dict["total_gross"], data_dict["total_net"]],
                 title="Gross vs Net Sales",
                 labels={'x': 'Sales Type', 'y': 'Amount'})
    return save_plotly_figure(fig, "profit_margin")


def calculate_number_of_transactions_data(sales_df):
    """Calculate the total transactions and group by period if possible."""
    if temp["Total Orders"] is None or temp["Total Orders"] not in sales_df.columns:
        return None
    total_transactions = sales_df[temp["Total Orders"]].sum()
    grouped_transactions = None
    if temp["Year"] in sales_df.columns and temp["Month"] in sales_df.columns:
        df = sales_df.copy()
        df["Period"] = pd.to_datetime(df[temp["Year"]].astype(str) + "-" +
                                      df[temp["Month"]].astype(str) + "-01")
        grouped_transactions = df.groupby("Period")[temp["Total Orders"]].sum().reset_index()
    return {"total_transactions": total_transactions, "grouped_transactions": grouped_transactions}

def plot_number_of_transactions(grouped_transactions):
    """Plot transactions over time as a line chart and save as HTML."""
    if grouped_transactions is None:
        return None
    fig = px.line(grouped_transactions, x="Period", y=temp["Total Orders"],
                  title="Transactions Over Time")
    return save_plotly_figure(fig, "transactions")


def calculate_peak_sales_period_data(sales_df):
    """Determine the period with peak sales."""
    if (temp["Year"] is None or temp["Year"] not in sales_df.columns or
        temp["Month"] is None or temp["Month"] not in sales_df.columns or
        temp["Total Sales"] is None or temp["Total Sales"] not in sales_df.columns):
        return None
    df = sales_df.copy()
    df["Period"] = pd.to_datetime(df[temp["Year"]].astype(str) + "-" +
                                  df[temp["Month"]].astype(str) + "-01")
    grouped = df.groupby("Period")[temp["Total Sales"]].sum()
    peak_day = grouped.idxmax()
    peak_value = grouped.max()
    grouped_df = grouped.reset_index()
    return {"grouped_data": grouped_df, "peak_day": peak_day, "peak_value": peak_value}

def plot_peak_sales_period(data_dict):
    """Plot sales over time with peak highlighted and save as HTML."""
    if data_dict is None:
        return None
    grouped_df = data_dict["grouped_data"]
    peak_day = data_dict["peak_day"]
    peak_value = data_dict["peak_value"]
    fig = px.line(grouped_df, x="Period", y=temp["Total Sales"],
                  title="Sales Over Time with Peak Highlighted")
    fig.add_scatter(x=[peak_day], y=[peak_value], mode="markers",
                    marker=dict(size=12, color="red"), name="Peak")
    return save_plotly_figure(fig, "peak_sales")


def calculate_seasonal_fluctuations_data(sales_df):
    """Calculate seasonal fluctuations (std. deviation) and grouped sales data."""
    if (temp["Year"] is None or temp["Year"] not in sales_df.columns or
        temp["Month"] is None or temp["Month"] not in sales_df.columns or
        temp["Total Sales"] is None or temp["Total Sales"] not in sales_df.columns):
        return None
    df = sales_df.copy()
    df["Period"] = pd.to_datetime(df[temp["Year"]].astype(str) + "-" +
                                  df[temp["Month"]].astype(str) + "-01")
    grouped = df.groupby("Period")[temp["Total Sales"]].sum()
    fluctuation = grouped.std()
    grouped_df = grouped.reset_index()
    return {"grouped_sales": grouped_df, "fluctuation": fluctuation}

def plot_seasonal_fluctuations(grouped_sales):
    """Plot seasonal sales fluctuations and save as HTML."""
    if grouped_sales is None:
        return None
    fig = px.line(grouped_sales, x="Period", y=temp["Total Sales"],
                  title="Seasonal Sales Fluctuations")
    return save_plotly_figure(fig, "seasonal_fluctuations")


def calculate_customer_churn_data(sales_df, reference_date=datetime.now()):
    """Calculate customer churn ratio based on last purchase date."""
    if (temp["customer_id"] is None or temp["customer_id"] not in sales_df.columns or
        temp["date"] is None or temp["date"] not in sales_df.columns):
        return None
    last_purchase = sales_df.groupby(temp["customer_id"])[temp["date"]].max()
    churned_customers = last_purchase[last_purchase < (reference_date - timedelta(days=30))].count()
    total_customers = sales_df[temp["customer_id"]].nunique()
    churn_ratio = churned_customers / total_customers if total_customers else None
    active_count = total_customers - churned_customers
    return {"churn_ratio": churn_ratio, "active_count": active_count, "churned_count": churned_customers}

def plot_customer_churn(active_count, churned_count):
    """Plot customer churn as a pie chart and save as HTML."""
    labels = ["Active", "Churned"]
    values = [active_count, churned_count]
    fig = px.pie(names=labels, values=values, title="Customer Churn")
    return save_plotly_figure(fig, "customer_churn")


def get_best_sellers_data(sales_df, top_n=3):
    """Identify the top-selling products."""
    if (temp["product_id"] is None or temp["product_id"] not in sales_df.columns or
        temp["quantity"] is None or temp["quantity"] not in sales_df.columns):
        return None
    product_sales = sales_df.groupby(temp["product_id"])[temp["quantity"]].sum().sort_values(ascending=False)
    best_sellers = product_sales.head(top_n)
    return best_sellers

def plot_best_sellers(best_sellers):
    """Plot best sellers as a bar chart and save as HTML."""
    if best_sellers is None:
        return None
    fig = px.bar(best_sellers.reset_index(), x=temp["product_id"], y=temp["quantity"],
                 title="Best Sellers")
    return save_plotly_figure(fig, "best_sellers")

def get_worst_sellers_data(sales_df, bottom_n=3):
    """Identify the worst-selling products."""
    if (temp["product_id"] is None or temp["product_id"] not in sales_df.columns or
        temp["quantity"] is None or temp["quantity"] not in sales_df.columns):
        return None
    product_sales = sales_df.groupby(temp["product_id"])[temp["quantity"]].sum().sort_values()
    worst_sellers = product_sales.head(bottom_n)
    return worst_sellers

def plot_worst_sellers(worst_sellers):
    """Plot worst sellers as a bar chart and save as HTML."""
    if worst_sellers is None:
        return None
    fig = px.bar(worst_sellers.reset_index(), x=temp["product_id"], y=temp["quantity"],
                 title="Worst Sellers")
    return save_plotly_figure(fig, "worst_sellers")


# =========================
# INVENTORY FUNCTIONS
# =========================

def get_stock_levels_data(inventory_df):
    """Extract stock level data from the inventory."""
    if (temp["product_id"] is None or temp["product_id"] not in inventory_df.columns or
        temp["stock_level"] is None or temp["stock_level"] not in inventory_df.columns):
        return None
    stock_df = inventory_df[[temp["product_id"], temp["stock_level"]]]
    return stock_df

def plot_stock_levels(stock_df):
    """Plot stock levels as a bar chart and save as HTML."""
    if stock_df is None:
        return None
    fig = px.bar(stock_df.sort_values(by=temp["stock_level"], ascending=False), 
                 x=temp["product_id"], y=temp["stock_level"], title="Stock Levels")
    return save_plotly_figure(fig, "stock_levels")


def forecast_stock_data(sales_df, inventory_df, days=30):
    """Forecast stock after a given number of days using average daily sales."""
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
    forecast_dict = {}
    for product in inventory_df[temp["product_id"]]:
        current_stock = inventory_df.loc[inventory_df[temp["product_id"]] == product, temp["stock_level"]].values[0]
        expected_sales = avg_daily_sales.get(product, 0) * days
        forecast_dict[product] = current_stock - expected_sales
    return forecast_dict

def plot_forecast_stock(forecast_dict, days=30):
    """Plot forecasted stock levels and save as HTML."""
    if forecast_dict is None:
        return None
    forecast_series = pd.Series(forecast_dict, name="Forecasted Stock")
    fig = px.bar(forecast_series.reset_index(), x="index", y="Forecasted Stock",
                 title=f"Forecast Stock After {days} Days",
                 labels={"index": "Product"})
    return save_plotly_figure(fig, "forecast_stock")


def suggest_stock_ordering_data(inventory_df, threshold=50, reorder_amount=100):
    """Suggest a reorder for products below the threshold."""
    if (temp["stock_level"] is None or temp["stock_level"] not in inventory_df.columns or
        temp["product_id"] is None or temp["product_id"] not in inventory_df.columns):
        return None
    orders = {}
    for _, row in inventory_df.iterrows():
        if row[temp["stock_level"]] < threshold:
            orders[row[temp["product_id"]]] = reorder_amount
    return orders

def plot_stock_ordering(orders):
    """Plot suggested stock orders as a bar chart and save as HTML."""
    if orders is None or len(orders) == 0:
        return None
    orders_series = pd.Series(orders, name="Reorder Amount")
    fig = px.bar(orders_series.reset_index(), x="index", y="Reorder Amount",
                 title="Suggested Stock Orders (Below Threshold)")
    return save_plotly_figure(fig, "stock_ordering")


def calculate_stock_valuation_data(inventory_df):
    """Calculate stock valuation as stock level * cost price."""
    if (temp["stock_level"] is None or temp["stock_level"] not in inventory_df.columns or
        temp["cost_price"] is None or temp["cost_price"] not in inventory_df.columns or
        temp["product_id"] is None or temp["product_id"] not in inventory_df.columns):
        return None
    df = inventory_df.copy()
    df['valuation'] = df[temp["stock_level"]] * df[temp["cost_price"]]
    valuation_df = df[[temp["product_id"], 'valuation']]
    return valuation_df

def plot_stock_valuation(valuation_df):
    """Plot stock valuation as a bar chart and save as HTML."""
    if valuation_df is None:
        return None
    fig = px.bar(valuation_df.sort_values(by='valuation', ascending=False), 
                 x=temp["product_id"], y="valuation", title="Stock Valuation")
    return save_plotly_figure(fig, "stock_valuation")


def check_stock_expiry_data(inventory_df, days=30, reference_date=datetime.now()):
    """Check which stock items will expire within a given number of days."""
    if (temp["expiry_date"] is None or temp["expiry_date"] not in inventory_df.columns or
        temp["product_id"] is None or temp["product_id"] not in inventory_df.columns):
        return None
    df = inventory_df.copy()
    df['days_to_expiry'] = (df[temp["expiry_date"]] - reference_date).dt.days
    expiring = df[df['days_to_expiry'] <= days]
    return expiring[[temp["product_id"], temp["expiry_date"], 'days_to_expiry']]

def plot_stock_expiry(expiring_df, days=30):
    """Plot stock items that are expiring soon and save as HTML."""
    if expiring_df is None or expiring_df.empty:
        return None
    fig = px.bar(expiring_df, x=temp["product_id"], y='days_to_expiry',
                 title=f"Stock Items Expiring Within {days} Days")
    return save_plotly_figure(fig, "stock_expiry")


def calculate_stock_spoilage_data(inventory_df):
    """Extract stock spoilage data.
       Note: Here we assume the inventory data has a column named 'spoilage'.
    """
    if (temp["product_id"] is None or temp["product_id"] not in inventory_df.columns or
        "spoilage" not in inventory_df.columns):
        return None
    spoilage_df = inventory_df[[temp["product_id"], "spoilage"]]
    return spoilage_df

def plot_stock_spoilage(spoilage_df):
    """Plot stock spoilage as a bar chart and save as HTML."""
    if spoilage_df is None:
        return None
    fig = px.bar(spoilage_df, x=temp["product_id"], y="spoilage",
                 title="Stock Spoilage")
    return save_plotly_figure(fig, "stock_spoilage")


def calculate_stock_returns_data(sales_df):
    """Calculate total returns from sales data."""
    if temp["Returns"] is None or temp["Returns"] not in sales_df.columns:
        return None
    total_returns = sales_df[temp["Returns"]].sum()
    return total_returns


# =========================
# ADVANCED FUNCTIONS
# =========================

def forecast_sales_prophet_data(sales_df, periods=30):
    """Forecast future sales using Prophet."""
    if (temp["Year"] is None or temp["Year"] not in sales_df.columns or
        temp["Month"] is None or temp["Month"] not in sales_df.columns or
        temp["Total Sales"] is None or temp["Total Sales"] not in sales_df.columns):
        return None
    df = sales_df.copy()
    df["Period"] = pd.to_datetime(df[temp["Year"]].astype(str) + "-" +
                                  df[temp["Month"]].astype(str) + "-01")
    grouped = df.groupby("Period")[temp["Total Sales"]].sum().reset_index()
    grouped.columns = ["ds", "y"]
    model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=False)
    model.fit(grouped)
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    return {"grouped": grouped, "forecast": forecast}

def plot_sales_prophet(forecast_data):
    """Plot Prophet forecast results and save as HTML."""
    if forecast_data is None:
        return None
    grouped = forecast_data["grouped"]
    forecast = forecast_data["forecast"]
    fig = px.line(forecast, x='ds', y='yhat', title="Prophet Forecast: Total Sales")
    fig.add_scatter(x=grouped['ds'], y=grouped['y'], mode='markers', name='Actual')
    return save_plotly_figure(fig, "sales_prophet")


def get_customer_demographics(sales_df):
    """Dummy implementation to create customer demographics data.
       In a real scenario, replace with actual demographics extraction.
    """
    customers = sales_df[temp["customer_id"]].unique()
    demographics = pd.DataFrame({
        temp["customer_id"]: customers,
        'gender': np.random.choice(['Male', 'Female'], size=len(customers)),
        'age': np.random.randint(18, 70, size=len(customers))
    })
    return demographics

def perform_customer_segmentation_data(sales_df):
    """Segment customers using basic clustering.
       Note: This dummy implementation creates transaction_value and transaction_id if missing.
    """
    if (temp["customer_id"] is None or temp["customer_id"] not in sales_df.columns):
        return None
    if "transaction_value" not in sales_df.columns:
        sales_df["transaction_value"] = np.random.rand(len(sales_df)) * 100
    if "transaction_id" not in sales_df.columns:
        sales_df["transaction_id"] = np.arange(len(sales_df))
    
    demographics = get_customer_demographics(sales_df)
    customer_sales = sales_df.groupby(temp["customer_id"]).agg(
        total_revenue=('transaction_value', 'sum'),
        transaction_count=('transaction_id', 'nunique')
    ).reset_index()
    df = pd.merge(demographics, customer_sales, on=temp["customer_id"], how='left').fillna(0)
    df['gender_numeric'] = df['gender'].map({'Male': 0, 'Female': 1})
    features = df[['age', 'total_revenue', 'transaction_count', 'gender_numeric']]
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(features)
    return df

def plot_customer_segmentation(segmented_df):
    """Plot customer segmentation results and save as HTML."""
    if segmented_df is None:
        return None
    fig = px.scatter(segmented_df, x='age', y='total_revenue', color='cluster',
                     hover_data=['transaction_count', 'gender'],
                     title="Customer Segmentation: Age vs Total Revenue")
    return save_plotly_figure(fig, "customer_segmentation")

def seasonal_decomposition_data(sales_df):
    """Perform seasonal decomposition on total sales."""
    if (temp["Year"] is None or temp["Year"] not in sales_df.columns or
        temp["Month"] is None or temp["Month"] not in sales_df.columns or
        temp["Total Sales"] is None or temp["Total Sales"] not in sales_df.columns):
        return None
    df = sales_df.copy()
    df["Period"] = pd.to_datetime(df[temp["Year"]].astype(str) + "-" +
                                  df[temp["Month"]].astype(str) + "-01")
    grouped = df.groupby("Period")[temp["Total Sales"]].sum().asfreq("MS").fillna(0)
    decomposition = seasonal_decompose(grouped, model="additive", period=12)
    return decomposition

def plot_seasonal_decomposition(decomposition):
    """Plot the components of seasonal decomposition using matplotlib."""
    if decomposition is None:
        return None
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    decomposition.observed.plot(ax=axes[0], title="Observed")
    decomposition.trend.plot(ax=axes[1], title="Trend")
    decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
    decomposition.resid.plot(ax=axes[3], title="Residual")
    plt.tight_layout()
    return save_plotly_figure(fig, "seasonal decomposition")

def correlation_heatmap_data(sales_df):
    """Compute the correlation matrix of numeric sales columns."""
    numeric_cols = sales_df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.any():
        return None
    corr = sales_df[numeric_cols].corr()
    return corr

def plot_correlation_heatmap(corr_matrix):
    """Plot the correlation matrix as a heatmap and save as HTML."""
    if corr_matrix is None:
        return None
    fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap: Sales Metrics")
    return save_plotly_figure(fig, "correlation_heatmap")

FUNCTIONS = {
    "calculate_total_revenue_data": calculate_total_revenue_data,
    "calculate_profit_margin_data": calculate_profit_margin_data,
    "calculate_number_of_transactions_data": calculate_number_of_transactions_data,
    "calculate_peak_sales_period_data": calculate_peak_sales_period_data,
    "calculate_seasonal_fluctuations_data": calculate_seasonal_fluctuations_data,
    "calculate_customer_churn_data": calculate_customer_churn_data,
    "get_best_sellers_data": get_best_sellers_data,
    "get_worst_sellers_data": get_worst_sellers_data,
    "get_stock_levels_data": get_stock_levels_data,
    "forecast_stock_data": forecast_stock_data,
    "suggest_stock_ordering_data": suggest_stock_ordering_data,
    "calculate_stock_valuation_data": calculate_stock_valuation_data,
    "check_stock_expiry_data": check_stock_expiry_data,
    "calculate_stock_returns_data": calculate_stock_returns_data,
    "forecast_sales_prophet_data": forecast_sales_prophet_data,
    "get_customer_demographics": get_customer_demographics,
    "perform_customer_segmentation_data": perform_customer_segmentation_data,
    "seasonal_decomposition_data": seasonal_decomposition_data
}

# =========================
# TESTING CODE
# =========================

# if __name__ == "__main__":
#     # Assume data is loaded from your JSON file.
#     data = pd.read_json('data_for_chroma/business.retailsales2.json')
    
#     # --- For testing sales-related functions ---
#     revenue_data = calculate_total_revenue_data(data)
#     if revenue_data is not None:
#         print("Total Revenue:", revenue_data["total_revenue"])
#         if revenue_data["grouped_data"] is not None:
#             rev_fig = plot_total_revenue(revenue_data["grouped_data"])
#             rev_fig.show()
    
#     profit_data = calculate_profit_margin_data(data)
#     if profit_data is not None:
#         print("Profit Margin:", profit_data["profit_margin"])
#         profit_fig = plot_profit_margin(profit_data)
#         profit_fig.show()
    
#     transactions_data = calculate_number_of_transactions_data(data)
#     if transactions_data is not None:
#         print("Total Transactions:", transactions_data["total_transactions"])
#         if transactions_data["grouped_transactions"] is not None:
#             trans_fig = plot_number_of_transactions(transactions_data["grouped_transactions"])
#             trans_fig.show()
    
#     peak_data = calculate_peak_sales_period_data(data)
#     if peak_data is not None:
#         print("Peak Sales Period:", peak_data["peak_day"], "with value", peak_data["peak_value"])
#         peak_fig = plot_peak_sales_period(peak_data)
#         peak_fig.show()
    
#     seasonal_data = calculate_seasonal_fluctuations_data(data)
#     if seasonal_data is not None:
#         print("Seasonal Sales Std Dev:", seasonal_data["fluctuation"])
#         season_fig = plot_seasonal_fluctuations(seasonal_data["grouped_sales"])
#         season_fig.show()
    
#     churn_data = calculate_customer_churn_data(data)
#     if churn_data is not None:
#         print("Customer Churn Ratio:", churn_data["churn_ratio"])
#         churn_fig = plot_customer_churn(churn_data["active_count"], churn_data["churned_count"])
#         churn_fig.show()
    
#     best_sellers = get_best_sellers_data(data)
#     if best_sellers is not None:
#         print("Best Sellers:\n", best_sellers)
#         best_fig = plot_best_sellers(best_sellers)
#         best_fig.show()
    
#     worst_sellers = get_worst_sellers_data(data)
#     if worst_sellers is not None:
#         print("Worst Sellers:\n", worst_sellers)
#         worst_fig = plot_worst_sellers(worst_sellers)
#         worst_fig.show()
    
#     # # --- For testing inventory-related functions ---
#     # # Create a dummy inventory DataFrame for testing
#     # data = pd.DataFrame({
#     #     # For testing, if temp keys are None then assign a name.
#     #     temp["product_id"] or "product_id": ['P1', 'P2', 'P3', 'P4'],
#     #     temp["stock_level"] or "stock_level": [100, 40, 60, 20],
#     #     temp["cost_price"] or "cost_price": [10, 20, 15, 5],
#     #     temp["expiry_date"] or "expiry_date": [datetime.now() + timedelta(days=15),
#     #                                            datetime.now() + timedelta(days=45),
#     #                                            datetime.now() + timedelta(days=10),
#     #                                            datetime.now() + timedelta(days=60)],
#     #     "spoilage": [5, 3, 2, 4]
#     # })
#     # # Set temp keys if not already set (for inventory functions)
#     # if temp["product_id"] is None:
#     #     temp["product_id"] = "product_id"
#     # if temp["stock_level"] is None:
#     #     temp["stock_level"] = "stock_level"
#     # if temp["cost_price"] is None:
#     #     temp["cost_price"] = "cost_price"
#     # if temp["expiry_date"] is None:
#     #     temp["expiry_date"] = "expiry_date"
    
#     stock_df = get_stock_levels_data(data)
#     if stock_df is not None:
#         stock_fig = plot_stock_levels(stock_df)
#         stock_fig.show()
    
#     forecast_dict = forecast_stock_data(data, data)
#     if forecast_dict is not None:
#         print("Forecasted Stock:", forecast_dict)
#         forecast_fig = plot_forecast_stock(forecast_dict)
#         forecast_fig.show()
    
#     orders = suggest_stock_ordering_data(data)
#     if orders is not None:
#         print("Suggested Orders:", orders)
#         orders_fig = plot_stock_ordering(orders)
#         if orders_fig:
#             orders_fig.show()
    
#     valuation_df = calculate_stock_valuation_data(data)
#     if valuation_df is not None:
#         print("Stock Valuation:\n", valuation_df)
#         valuation_fig = plot_stock_valuation(valuation_df)
#         valuation_fig.show()
    
#     expiring_df = check_stock_expiry_data(data, days=30)
#     if expiring_df is not None:
#         print("Expiring Stock Items:\n", expiring_df)
#         expiry_fig = plot_stock_expiry(expiring_df, days=30)
#         expiry_fig.show()
    
#     spoilage_df = calculate_stock_spoilage_data(data)
#     if spoilage_df is not None:
#         print("Stock Spoilage:\n", spoilage_df)
#         spoilage_fig = plot_stock_spoilage(spoilage_df)
#         spoilage_fig.show()
    
#     returns_total = calculate_stock_returns_data(data)
#     if returns_total is not None:
#         print("Total Returns:", returns_total)
#         returns_fig = plot_stock_returns(returns_total)
#         returns_fig.show()
    
#     # --- Advanced functions tests ---
#     prophet_data = forecast_sales_prophet_data(data, periods=10)
#     if prophet_data is not None:
#         prophet_fig = plot_sales_prophet(prophet_data)
#         prophet_fig.show()
    
#     segmentation_df = perform_customer_segmentation_data(data)
#     if segmentation_df is not None:
#         segmentation_fig = plot_customer_segmentation(segmentation_df)
#         segmentation_fig.show()
    
#     decomposition = seasonal_decomposition_data(data)
#     if decomposition is not None:
#         decomp_fig = plot_seasonal_decomposition(decomposition)
#         plt.show()  # Use matplotlib's show() for the decomposition plots
    
#     corr_matrix = correlation_heatmap_data(data)
#     if corr_matrix is not None:
#         heatmap_fig = plot_correlation_heatmap(corr_matrix)
#         heatmap_fig.show()
