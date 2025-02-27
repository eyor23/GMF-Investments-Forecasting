import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

def load_and_clean_data(tickers, start_date, end_date):
    """Loads and cleans financial data for given tickers."""
    data_dict = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data is not None and not data.empty:
            # Flatten MultiIndex columns
            data.columns = [col[0] for col in data.columns]
            #Use ffill() instead of fillna(method='ffill')
            data.ffill(inplace=True)
            data.index = pd.to_datetime(data.index)
            data_dict[ticker] = data
        else:
            print(f"No data found for {ticker}")
    return data_dict

def get_date_range(year_start = 2015, month_start= 1, day_start = 1, year_end = 2025, month_end = 1, day_end = 31):
    """Returns start and end datetime objects."""
    start_date = datetime.datetime(year_start, month_start, day_start)
    end_date = datetime.datetime(year_end, month_end, day_end)
    return start_date, end_date

def plot_time_series(data, asset_name, column="Close"):
    """Plots the time series for a given column."""
    plt.figure(figsize=(12, 6))
    plt.plot(data[column])
    plt.title(f"{asset_name} {column} Over Time")
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.grid(True)
    plt.show()

def plot_daily_returns(data, asset_name):
    """Plots the daily percentage change."""
    daily_returns = data["Close"].pct_change().dropna()
    plt.figure(figsize=(12, 6))
    plt.plot(daily_returns)
    plt.title(f"{asset_name} Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.grid(True)
    plt.show()

def plot_rolling_stats(data, asset_name, window=20):
    """Plots rolling mean and standard deviation."""
    rolling_mean = data["Close"].rolling(window=window).mean()
    rolling_std = data["Close"].rolling(window=window).std()
    plt.figure(figsize=(12, 6))
    plt.plot(data["Close"], label="Close")
    plt.plot(rolling_mean, label=f"{window}-Day Rolling Mean")
    plt.plot(rolling_std, label=f"{window}-Day Rolling Std")
    plt.title(f"{asset_name} Rolling Mean and Std")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_outliers(daily_returns, asset_name):
    """Plots a box plot to visualize outliers."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=daily_returns)
    plt.title(f"{asset_name} Daily Returns Box Plot")
    plt.show()

def analyze_high_low_returns(daily_returns, asset_name, n=5):
    """Prints the highest and lowest return days."""
    print(f"\n{asset_name} Highest Returns:")
    print(daily_returns.nlargest(n))
    print(f"\n{asset_name} Lowest Returns:")
    print(daily_returns.nsmallest(n))

def decompose_time_series(data, asset_name, period=252):
    """Decomposes the time series into trend, seasonal, and residual components."""
    decomposition = seasonal_decompose(data["Close"], model="additive", period=period)
    plt.figure(figsize=(16, 12))
    decomposition.plot()
    plt.suptitle(f"{asset_name} Time Series Decomposition", fontsize=16)
    plt.show()

def calculate_var_sharpe(daily_returns, asset_name, confidence_level=0.95, risk_free_rate=0.02):
    """Calculates VaR and Sharpe Ratio."""
    var = daily_returns.quantile(1 - confidence_level)
    print(f"\n{asset_name} Value at Risk (VaR) at {confidence_level*100}%: {var}")
    sharpe_ratio = (daily_returns.mean() - risk_free_rate/252) / daily_returns.std()
    print(f"{asset_name} Sharpe Ratio: {sharpe_ratio}")