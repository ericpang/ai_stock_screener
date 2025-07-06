# indicators.py
import ta
import pandas as pd

import pandas as pd
import ta

def add_indicators(data_dict):
    processed = {}
    for ticker, df in data_dict.items():
        df = df.copy()

        try:
            # Ensure 'Close' and 'Volume' are 1D Series (not accidental 2D from slicing)
            if isinstance(df['Close'], pd.DataFrame):
                df['Close'] = df['Close'].squeeze()
            if isinstance(df['Volume'], pd.DataFrame):
                df['Volume'] = df['Volume'].squeeze()

            df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
            df['SMA50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
            df['SMA200'] = ta.trend.SMAIndicator(close=df['Close'], window=200).sma_indicator()
            df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()

            processed[ticker] = df
            #print(f"{ticker} → columns: {df.columns.tolist()}")
            #print(df[['RSI', 'SMA50', 'SMA200', 'Volume_SMA20']].tail())

        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")

    return processed

'''
def add_indicators(data_dict):
    processed = {}
    for ticker, df in data_dict.items():
        df = df.copy()
        try:
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            df['SMA50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
            df['SMA200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
            df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
            processed[ticker] = df
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    return processed
'''