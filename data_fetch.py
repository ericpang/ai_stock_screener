import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
#from config import IB_GATEWAY_HOST, IB_GATEWAY_PORT, CLIENT_ID, DEFAULT_DURATION, DEFAULT_BAR_SIZE

'''This script fetches live market data for S&P 500 stocks using yfinance,

# Load S&P 500 tickers (can also use ETF if needed)
def get_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    return table[0]['Symbol'].tolist()
adds technical indicators, and prepares the data for further analysis or modeling.'''

def fetch_ticker_data(ticker, period="6mo"):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    df.reset_index(inplace=True)  # Ensure 'Date' is a column
    return df

# Get live market data for a list of tickers
def fetch_all_tickers(tickers, period='1y', interval='1d'):
    all_data = {}
    for ticker in tqdm(tickers, desc="Fetching ticker data"):
    #for ticker in tickers:
        try:
            #stock = yf.Ticker(ticker)
            #hist = stock.history(period=period, interval=interval)
            #pe_ratio = stock.info.get('trailingPE', None)

            hist = yf.download(ticker, period=period, interval=interval, progress=False,  auto_adjust=True)

            # Flatten MultiIndex columns like ('Close', 'MMM') → 'Close'
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.droplevel(1)

            pe_ratio = None
            try:
                pe_ratio = yf.Ticker(ticker).info.get('trailingPE', None)
            except:
                pass            

            if not hist.empty:
                #hist.reset_index(inplace=True)
                hist['PE'] = pe_ratio
                all_data[ticker] = hist

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    return all_data


'''

# Example usage
tickers = get_sp500_tickers()[:3]  # limit to 50 for testing
stock_data = fetch_stock_data(tickers)

# Check one sample
#print(stock_data['AAPL'].tail())



# Function to add indicators to the historical data
def add_indicators(data_dict):
    processed = {}
    for ticker, df in data_dict.items():
        df = df.copy()
        try:
            # Add RSI
            df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
            
            # Add SMA 50 and 200
            df['SMA50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
            df['SMA200'] = ta.trend.SMAIndicator(close=df['Close'], window=200).sma_indicator()

            # Add volume trend (SMA of volume)
            df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()

            processed[ticker] = df
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    return processed

# Apply the function
stock_data_with_indicators = add_indicators(stock_data)

# Preview sample
print(stock_data_with_indicators['MMM'][['Close', 'RSI', 'SMA50', 'SMA200', 'Volume', 'Volume_SMA20']].tail())
'''