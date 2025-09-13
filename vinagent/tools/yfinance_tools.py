import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vinagent.register import primary_function


@primary_function
def fetch_stock_data(
    symbol: str,
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.

    Args:
        symbol (str): The stock symbol (e.g., 'AAPL' for Apple).
        start_date (str): Start date for historical data (YYYY-MM-DD).
        end_date (str): End date for historical data (YYYY-MM-DD).
        interval (str): Data interval ('1d', '1wk', '1mo', etc.).

    Returns:
        pd.DataFrame: DataFrame containing historical stock prices.
    """
    try:
        # limiter = Limiter(history_rate)
        # session = LimiterSession(limiter=limiter)
        # session.headers['User-agent'] = 'tickerpicker/1.0'
        # stock = yf.Ticker(symbol, session=session)
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date, interval=interval)
        if data.empty:
            print(f"No data found for {symbol}. Check the symbol or date range.")
            return None
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


@primary_function
def visualize_stock_data(
    symbol: str,
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
    interval: str = "1d",
) -> None:
    """
    Visualize stock data with multiple chart types.

    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD). It must be greater than start_date.
        interval (str): Data interval ('1d', '1wk', '1mo')
    """
    # Fetch the data
    df = fetch_stock_data(symbol, start_date, end_date, interval)
    if df is None:
        return

    # Reset index for easier plotting
    df = df.reset_index()

    # 1. Matplotlib - Price and Volume Plot
    plt.figure(figsize=(12, 8))

    # Price subplot
    plt.subplot(2, 1, 1)
    plt.plot(df["Date"], df["Close"], label="Close Price", color="blue")
    plt.title(f"{symbol} Stock Price and Volume")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)

    # Volume subplot
    plt.subplot(2, 1, 2)
    plt.bar(df["Date"], df["Volume"], color="gray")
    plt.ylabel("Volume")
    plt.xlabel("Date")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 2. Plotly - Interactive Candlestick Chart with Moving Average
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Candlestick", "Volume"),
        row_heights=[0.7, 0.3],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    # 20-day Moving Average
    df["MA20"] = df["Close"].rolling(window=20).mean()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA20"],
            line=dict(color="purple", width=1),
            name="20-day MA",
        ),
        row=1,
        col=1,
    )

    # Volume
    fig.add_trace(
        go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color="gray"),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        title=f"{symbol} Stock Price Analysis",
        yaxis_title="Price ($)",
        height=800,
        showlegend=True,
        template="plotly_white",
    )

    # Update axes
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.show()
    return fig


@primary_function
def plot_returns(
    symbol: str,
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
    interval: str = "1d",
) -> None:
    """
    Visualize cumulative returns of the stock.
    """
    df = fetch_stock_data(symbol, start_date, end_date, interval)
    if df is None:
        return

    # Calculate daily returns and cumulative returns
    df["Daily_Return"] = df["Close"].pct_change()
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1

    # Plot with Plotly
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Cumulative_Return"] * 100,
            mode="lines",
            name="Cumulative Return",
            line=dict(color="green"),
        )
    )

    fig.update_layout(
        title=f"{symbol} Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        template="plotly_white",
        height=500,
    )

    fig.show()
    return fig
