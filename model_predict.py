# model_predict.py

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_model(model_path='models/xgb_model.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def prepare_features(df):
    features = df[['RSI', 'SMA50', 'SMA200', 'Volume_SMA20']].copy()
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features

def predict_ticker_latest(model, df, ticker):
    if not isinstance(df, pd.DataFrame):
        print(f"⚠️ Skipping {ticker}: Not a DataFrame ({type(df)})")
        return None, None

    df = df.dropna(subset=['RSI', 'SMA50', 'SMA200', 'Volume_SMA20'])
    if df.empty:
        return None, None

    latest = df.iloc[-1:][['RSI', 'SMA50', 'SMA200', 'Volume_SMA20']]
    if latest.isnull().any(axis=1).iloc[0]:
        return None, None

    prob = model.predict_proba(latest)[0][1]
    result = {
        'Ticker': ticker,
        'RSI': latest.iloc[0]['RSI'],
        'SMA50': latest.iloc[0]['SMA50'],
        'SMA200': latest.iloc[0]['SMA200'],
        'Volume_SMA20': latest.iloc[0]['Volume_SMA20'],
        'Probability': prob
    }
    return result, df

def predict_all(model, processed_data_dict, rsi_range=(0, 100), min_prob=0.6, generate_charts=False):
    """
    Predicts all tickers and optionally filters by RSI/probability
    Returns (DataFrame, {ticker: df}, {ticker: chart_fig})
    """
    results = []
    valid_dfs = {}
    chart_figures = {}

    for ticker, df in processed_data_dict.items():
        try:
            if not isinstance(df, pd.DataFrame):
                print(f"⚠️ Skipping {ticker}: Not a DataFrame ({type(df)})")
                continue

            result, full_df = predict_ticker_latest(model, df, ticker)
            if result is None:
                continue

            if not (rsi_range[0] <= result['RSI'] <= rsi_range[1]):
                continue
            if result['Probability'] < min_prob:
                continue

            results.append(result)
            valid_dfs[ticker] = full_df

            if generate_charts:
                #print(f"{ticker} chart data: {full_df.index.min()} → {full_df.index.max()}")
                #print(full_df.tail(3)[['Date', 'RSI', 'SMA50', 'SMA200', 'Volume_SMA20']])                
                chart_figures[ticker] = plot_chart(full_df, ticker, rsi_range)

        except Exception as e:
            print(f"Prediction failed for {ticker}: {e}")


    if not results:
        return pd.DataFrame(), {}, {}

    df_result = pd.DataFrame(results).sort_values(by='Probability', ascending=False)
    return df_result, valid_dfs, chart_figures

    #df_result = pd.DataFrame(results).sort_values(by='Probability', ascending=False)
    #return df_result, valid_dfs, chart_figures

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Cursor

def plot_chart(df, ticker, rsi_range=(30, 70)):
    df = df.copy()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    df = df.dropna(subset=['Close', 'SMA50', 'SMA200', 'RSI'])

    if df.empty:
        print(f"⚠️ {ticker} has no valid data to plot.")
        return None

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # --- Price chart
    ax[0].plot(df.index, df['Close'], label='Close', color='black')
    ax[0].plot(df.index, df['SMA50'], label='SMA50', linestyle='--', color='blue')
    ax[0].plot(df.index, df['SMA200'], label='SMA200', linestyle='--', color='orange')
    ax[0].set_ylabel('Price', fontsize=8) # Smaller font for y-axis label
    ax[0].legend()
    ax[0].set_title(f'{ticker} Price & RSI')
    # Ensure the crosshair is visible
    #ax[0].autoscale(enable=True, axis='both', tight=False)

    # --- RSI chart
    ax[1].plot(df.index, df['RSI'], label='RSI', color='purple')
    ax[1].axhline(rsi_range[1], color='red', linestyle='--', linewidth=1)
    ax[1].axhline(rsi_range[0], color='green', linestyle='--', linewidth=1)
    ax[1].set_ylabel('RSI', fontsize=8) # Smaller font for y-axis label
    ax[1].legend()

    # ✅ Format x-axis with daily ticks
    locator = mdates.DayLocator()
    formatter = mdates.DateFormatter('%Y-%m-%d')
    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)
    # Ensure the crosshair is visible
    #ax[1].autoscale(enable=True, axis='both', tight=False)

    for a in ax:
        a.tick_params(axis='x', labelrotation=45, labelsize=8) # Smaller font for x-axis ticks
        a.tick_params(axis='y', labelsize=8) # Smaller font for y-axis ticks

    fig.tight_layout()

    # --- Add crosshair indicator
    # Create a Cursor object for each subplot
    cursor1 = Cursor(ax[0], horizOn=True, vertOn=True, color='gray', linewidth=0.8)
    cursor2 = Cursor(ax[1], horizOn=True, vertOn=True, color='gray', linewidth=0.8)

    # You need to keep a reference to the cursors to prevent them from being garbage collected
    # In a typical script, if you call `plt.show()`, these will persist.
    # If this is part of a larger application, you might store them as attributes
    # of a class or in a list if you have many plots.
    fig._cursors = [cursor1, cursor2] # Storing them as an attribute of the figure

    return fig

'''
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

def plot_chart(df, ticker, rsi_range=(30, 70)):
    df = df.copy()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    df = df.dropna(subset=['Close', 'SMA50', 'SMA200', 'RSI'])

    if df.empty:
        print(f"⚠️ {ticker} has no valid data to plot.")
        return None

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # --- Price chart
    ax[0].plot(df.index, df['Close'], label='Close', color='black')
    ax[0].plot(df.index, df['SMA50'], label='SMA50', linestyle='--', color='blue')
    ax[0].plot(df.index, df['SMA200'], label='SMA200', linestyle='--', color='orange')
    ax[0].set_ylabel('Price')
    ax[0].legend()
    ax[0].set_title(f'{ticker} Price & RSI')

    # --- RSI chart
    ax[1].plot(df.index, df['RSI'], label='RSI', color='purple')
    ax[1].axhline(rsi_range[1], color='red', linestyle='--', linewidth=1)
    ax[1].axhline(rsi_range[0], color='green', linestyle='--', linewidth=1)
    ax[1].set_ylabel('RSI')
    ax[1].legend()

    # ✅ Format x-axis with daily ticks
    locator = mdates.DayLocator()
    formatter = mdates.DateFormatter('%Y-%m-%d')
    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

    for a in ax:
        a.tick_params(axis='x', labelrotation=45)

    fig.tight_layout()
    return fig

'''