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

# Display the first few rows
# data.head()

# =========================
# UTILITY FUNCTION (UNCHANGED)
# =========================

def save_plotly_figure(fig, plot_name_prefix):
    """
    Save the given Plotly figure as a standalone HTML file in the media/plots folder.
    Returns the URL to the saved file.
    """
    plot_dir = os.path.join(settings.MEDIA_ROOT, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{plot_name_prefix}_{timestamp}.html"
    file_path = os.path.join(plot_dir, file_name)
    fig.write_html(file_path)
    file_url = os.path.join(settings.MEDIA_URL, "plots", file_name)
    return file_url


# =========================
# SALES FUNCTIONS (MERGED)
# =========================

def calculate_total_revenue_data(sales_df):
    """Calculate total revenue and group by period if possible, and plot the result."""
    if temp["Total Sales"] is None or temp["Total Sales"] not in sales_df.columns:
        return None
    total_revenue = sales_df[temp["Total Sales"]].sum()
    grouped_data = None
    if temp["Year"] in sales_df.columns and temp["Month"] in sales_df.columns:
        df = sales_df.copy()
        df["Period"] = pd.to_datetime(df[temp["Year"]].astype(str) + "-" + 
                                      df[temp["Month"]].astype(str) + "-01")
        grouped_data = df.groupby("Period")[temp["Total Sales"]].sum().reset_index()
    plot_url = None
    if grouped_data is not None:
        fig = px.bar(grouped_data, x="Period", y=temp["Total Sales"],
                     title="Total Revenue Over Time")
        plot_url = save_plotly_figure(fig, "total_revenue")
    return {"total_revenue": total_revenue, "grouped_data": grouped_data, "plot_url": plot_url}


def calculate_profit_margin_data(sales_df):
    """Compute gross and net sales, calculate profit margin, and plot the comparison."""
    if (temp["Gross Sales"] is None or temp["Gross Sales"] not in sales_df.columns or
        temp["Net Sales"] is None or temp["Net Sales"] not in sales_df.columns):
        return None
    total_gross = sales_df[temp["Gross Sales"]].sum()
    total_net = sales_df[temp["Net Sales"]].sum()
    profit_margin = total_net / total_gross if total_gross > 0 else None
    fig = px.bar(x=["Gross Sales", "Net Sales"],
                 y=[total_gross, total_net],
                 title="Gross vs Net Sales",
                 labels={'x': 'Sales Type', 'y': 'Amount'})
    plot_url = save_plotly_figure(fig, "profit_margin")
    return {"total_gross": total_gross, "total_net": total_net, "profit_margin": profit_margin, "plot_url": plot_url}


def calculate_number_of_transactions_data(sales_df):
    """Calculate total transactions and group by period if possible, and plot the trend."""
    if temp["Total Orders"] is None or temp["Total Orders"] not in sales_df.columns:
        return None
    total_transactions = sales_df[temp["Total Orders"]].sum()
    grouped_transactions = None
    if temp["Year"] in sales_df.columns and temp["Month"] in sales_df.columns:
        df = sales_df.copy()
        df["Period"] = pd.to_datetime(df[temp["Year"]].astype(str) + "-" +
                                      df[temp["Month"]].astype(str) + "-01")
        grouped_transactions = df.groupby("Period")[temp["Total Orders"]].sum().reset_index()
    plot_url = None
    if grouped_transactions is not None:
        fig = px.line(grouped_transactions, x="Period", y=temp["Total Orders"],
                      title="Transactions Over Time")
        plot_url = save_plotly_figure(fig, "transactions")
    return {"total_transactions": total_transactions, "grouped_transactions": grouped_transactions, "plot_url": plot_url}


def calculate_peak_sales_period_data(sales_df):
    """Determine the period with peak sales and plot the result."""
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
    fig = px.line(grouped_df, x="Period", y=temp["Total Sales"],
                  title="Sales Over Time with Peak Highlighted")
    fig.add_scatter(x=[peak_day], y=[peak_value], mode="markers",
                    marker=dict(size=12, color="red"), name="Peak")
    plot_url = save_plotly_figure(fig, "peak_sales")
    return {"grouped_data": grouped_df, "peak_day": peak_day, "peak_value": peak_value, "plot_url": plot_url}


def calculate_seasonal_fluctuations_data(sales_df):
    """Calculate seasonal fluctuations (std. deviation) and plot the grouped sales."""
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
    fig = px.line(grouped_df, x="Period", y=temp["Total Sales"],
                  title="Seasonal Sales Fluctuations")
    plot_url = save_plotly_figure(fig, "seasonal_fluctuations")
    return {"grouped_sales": grouped_df, "fluctuation": fluctuation, "plot_url": plot_url}


def calculate_customer_churn_data(sales_df, reference_date=datetime.now()):
    """Calculate customer churn ratio based on last purchase date and plot the churn pie chart."""
    if (temp["customer_id"] is None or temp["customer_id"] not in sales_df.columns or
        temp["date"] is None or temp["date"] not in sales_df.columns):
        return None
    last_purchase = sales_df.groupby(temp["customer_id"])[temp["date"]].max()
    churned_customers = last_purchase[last_purchase < (reference_date - timedelta(days=30))].count()
    total_customers = sales_df[temp["customer_id"]].nunique()
    churn_ratio = churned_customers / total_customers if total_customers else None
    active_count = total_customers - churned_customers
    fig = px.pie(names=["Active", "Churned"], values=[active_count, churned_customers],
                 title="Customer Churn")
    plot_url = save_plotly_figure(fig, "customer_churn")
    return {"churn_ratio": churn_ratio, "active_count": active_count, "churned_count": churned_customers, "plot_url": plot_url}


def get_best_sellers_data(sales_df, top_n=3):
    """Identify the top-selling products and plot them."""
    if (temp["product_id"] is None or temp["product_id"] not in sales_df.columns or
        temp["quantity"] is None or temp["quantity"] not in sales_df.columns):
        return None
    product_sales = sales_df.groupby(temp["product_id"])[temp["quantity"]].sum().sort_values(ascending=False)
    best_sellers = product_sales.head(top_n)
    fig = px.bar(best_sellers.reset_index(), x=temp["product_id"], y=temp["quantity"],
                 title="Best Sellers")
    plot_url = save_plotly_figure(fig, "best_sellers")
    return {"best_sellers": best_sellers, "plot_url": plot_url}


def get_worst_sellers_data(sales_df, bottom_n=3):
    """Identify the worst-selling products and plot them."""
    if (temp["product_id"] is None or temp["product_id"] not in sales_df.columns or
        temp["quantity"] is None or temp["quantity"] not in sales_df.columns):
        return None
    product_sales = sales_df.groupby(temp["product_id"])[temp["quantity"]].sum().sort_values()
    worst_sellers = product_sales.head(bottom_n)
    fig = px.bar(worst_sellers.reset_index(), x=temp["product_id"], y=temp["quantity"],
                 title="Worst Sellers")
    plot_url = save_plotly_figure(fig, "worst_sellers")
    return {"worst_sellers": worst_sellers, "plot_url": plot_url}


# =========================
# INVENTORY FUNCTIONS (MERGED)
# =========================

def get_stock_levels_data(inventory_df):
    """Extract stock level data from the inventory and plot it."""
    if (temp["product_id"] is None or temp["product_id"] not in inventory_df.columns or
        temp["stock_level"] is None or temp["stock_level"] not in inventory_df.columns):
        return None
    stock_df = inventory_df[[temp["product_id"], temp["stock_level"]]]
    fig = px.bar(stock_df.sort_values(by=temp["stock_level"], ascending=False), 
                 x=temp["product_id"], y=temp["stock_level"], title="Stock Levels")
    plot_url = save_plotly_figure(fig, "stock_levels")
    return {"stock_levels": stock_df, "plot_url": plot_url}


def forecast_stock_data(sales_df, inventory_df, days=30):
    """Forecast stock after a given number of days using average daily sales and plot the forecast."""
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
    fig = px.bar(pd.Series(forecast_dict, name="Forecasted Stock").reset_index(), 
                 x="index", y="Forecasted Stock",
                 title=f"Forecast Stock After {days} Days",
                 labels={"index": "Product"})
    plot_url = save_plotly_figure(fig, "forecast_stock")
    return {"forecast": forecast_dict, "plot_url": plot_url}


def suggest_stock_ordering_data(inventory_df, threshold=50, reorder_amount=100):
    """Suggest a reorder for products below the threshold and plot the suggested orders."""
    if (temp["stock_level"] is None or temp["stock_level"] not in inventory_df.columns or
        temp["product_id"] is None or temp["product_id"] not in inventory_df.columns):
        return None
    orders = {}
    for _, row in inventory_df.iterrows():
        if row[temp["stock_level"]] < threshold:
            orders[row[temp["product_id"]]] = reorder_amount
    fig = None
    if orders:
        orders_series = pd.Series(orders, name="Reorder Amount")
        fig = px.bar(orders_series.reset_index(), x="index", y="Reorder Amount",
                     title="Suggested Stock Orders (Below Threshold)")
    plot_url = save_plotly_figure(fig, "stock_ordering") if fig is not None else None
    return {"orders": orders, "plot_url": plot_url}


def calculate_stock_valuation_data(inventory_df):
    """Calculate stock valuation as stock level * cost price and plot it."""
    if (temp["stock_level"] is None or temp["stock_level"] not in inventory_df.columns or
        temp["cost_price"] is None or temp["cost_price"] not in inventory_df.columns or
        temp["product_id"] is None or temp["product_id"] not in inventory_df.columns):
        return None
    df = inventory_df.copy()
    df['valuation'] = df[temp["stock_level"]] * df[temp["cost_price"]]
    valuation_df = df[[temp["product_id"], 'valuation']]
    fig = px.bar(valuation_df.sort_values(by='valuation', ascending=False), 
                 x=temp["product_id"], y="valuation", title="Stock Valuation")
    plot_url = save_plotly_figure(fig, "stock_valuation")
    return {"valuation": valuation_df, "plot_url": plot_url}


def check_stock_expiry_data(inventory_df, days=30, reference_date=datetime.now()):
    """Check which stock items will expire within a given number of days and plot them."""
    if (temp["expiry_date"] is None or temp["expiry_date"] not in inventory_df.columns or
        temp["product_id"] is None or temp["product_id"] not in inventory_df.columns):
        return None
    df = inventory_df.copy()
    df['days_to_expiry'] = (df[temp["expiry_date"]] - reference_date).dt.days
    expiring = df[df['days_to_expiry'] <= days]
    fig = None
    plot_url = None
    if not expiring.empty:
        fig = px.bar(expiring, x=temp["product_id"], y='days_to_expiry',
                     title=f"Stock Items Expiring Within {days} Days")
        plot_url = save_plotly_figure(fig, "stock_expiry")
    return {"expiring": expiring[[temp["product_id"], temp["expiry_date"], 'days_to_expiry']], "plot_url": plot_url}


def calculate_stock_spoilage_data(inventory_df):
    """Extract stock spoilage data and plot it.
       Note: Assumes the inventory data has a column named 'spoilage'.
    """
    if (temp["product_id"] is None or temp["product_id"] not in inventory_df.columns or
        "spoilage" not in inventory_df.columns):
        return None
    spoilage_df = inventory_df[[temp["product_id"], "spoilage"]]
    fig = px.bar(spoilage_df, x=temp["product_id"], y="spoilage",
                 title="Stock Spoilage")
    plot_url = save_plotly_figure(fig, "stock_spoilage")
    return {"spoilage": spoilage_df, "plot_url": plot_url}


def calculate_stock_returns_data(sales_df):
    """Calculate total returns from sales data. (No plotting provided)"""
    if temp["Returns"] is None or temp["Returns"] not in sales_df.columns:
        return None
    total_returns = sales_df[temp["Returns"]].sum()
    return total_returns


# =========================
# ADVANCED FUNCTIONS (MERGED)
# =========================

def forecast_sales_prophet_data(sales_df, periods=30):
    """Forecast future sales using Prophet and plot the forecast."""
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
    fig = px.line(forecast, x='ds', y='yhat', title="Prophet Forecast: Total Sales")
    fig.add_scatter(x=grouped['ds'], y=grouped['y'], mode='markers', name='Actual')
    plot_url = save_plotly_figure(fig, "sales_prophet")
    return {"grouped": grouped, "forecast": forecast, "plot_url": plot_url}


def perform_customer_segmentation_data(sales_df):
    """Segment customers using basic clustering and plot the segmentation."""
    if (temp["customer_id"] is None or temp["customer_id"] not in sales_df.columns):
        return None
    if "transaction_value" not in sales_df.columns:
        sales_df["transaction_value"] = np.random.rand(len(sales_df)) * 100
    if "transaction_id" not in sales_df.columns:
        sales_df["transaction_id"] = np.arange(len(sales_df))
    
    customers = sales_df[temp["customer_id"]].unique()
    demographics = pd.DataFrame({
        temp["customer_id"]: customers,
        'gender': np.random.choice(['Male', 'Female'], size=len(customers)),
        'age': np.random.randint(18, 70, size=len(customers))
    })
    customer_sales = sales_df.groupby(temp["customer_id"]).agg(
        total_revenue=('transaction_value', 'sum'),
        transaction_count=('transaction_id', 'nunique')
    ).reset_index()
    df = pd.merge(demographics, customer_sales, on=temp["customer_id"], how='left').fillna(0)
    df['gender_numeric'] = df['gender'].map({'Male': 0, 'Female': 1})
    features = df[['age', 'total_revenue', 'transaction_count', 'gender_numeric']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(features)
    fig = px.scatter(df, x='age', y='total_revenue', color='cluster',
                     hover_data=['transaction_count', 'gender'],
                     title="Customer Segmentation: Age vs Total Revenue")
    plot_url = save_plotly_figure(fig, "customer_segmentation")
    return {"segmentation": df, "plot_url": plot_url}


def seasonal_decomposition_data(sales_df):
    """Perform seasonal decomposition on total sales and plot the components."""
    if (temp["Year"] is None or temp["Year"] not in sales_df.columns or
        temp["Month"] is None or temp["Month"] not in sales_df.columns or
        temp["Total Sales"] is None or temp["Total Sales"] not in sales_df.columns):
        return None
    df = sales_df.copy()
    df["Period"] = pd.to_datetime(df[temp["Year"]].astype(str) + "-" +
                                  df[temp["Month"]].astype(str) + "-01")
    grouped = df.groupby("Period")[temp["Total Sales"]].sum().asfreq("MS").fillna(0)
    decomposition = seasonal_decompose(grouped, model="additive", period=12)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    decomposition.observed.plot(ax=axes[0], title="Observed")
    decomposition.trend.plot(ax=axes[1], title="Trend")
    decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
    decomposition.resid.plot(ax=axes[3], title="Residual")
    plt.tight_layout()
    plot_url = save_plotly_figure(fig, "seasonal decomposition")
    return {"decomposition": decomposition, "plot_url": plot_url}


def correlation_heatmap_data(sales_df):
    """Compute the correlation matrix of numeric sales columns and plot the heatmap."""
    numeric_cols = sales_df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.any():
        return None
    corr = sales_df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap: Sales Metrics")
    plot_url = save_plotly_figure(fig, "correlation_heatmap")
    return {"correlation_matrix": corr, "plot_url": plot_url}

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
    "calculate_stock_returns_data": calculate_stock_returns_data,  # No plotting function provided
    "forecast_sales_prophet_data": forecast_sales_prophet_data,
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
