# generate_sp500_symbols.py
import pandas as pd

def save_sp500_symbols(output_path='data/symbols_sp500.csv'):
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    symbols = tables[0]['Symbol']
    symbols.to_csv(output_path, index=False)
    print(f"Saved {len(symbols)} symbols to {output_path}")

if __name__ == "__main__":
    save_sp500_symbols()
