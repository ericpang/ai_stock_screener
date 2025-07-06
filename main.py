# main.py
import pandas as pd
from data_fetch import fetch_all_tickers
from indicators import add_indicators
from model_train import prepare_dataset, train_and_save_model
from model_predict import load_model, predict_all

'''Main script to fetch S&P 500 data, add indicators, prepare dataset, and train model.
import os

# save S&P 500 symbols if not already present
if not os.path.exists('data/symbols_sp500.csv'):
    from generate_sp500_symbols import save_sp500_symbols
    save_sp500_symbols()
'''



# 💡 Set thresholds dynamically or from config
threshold = 0.10
future_days = 30




# Load S&P 500 symbols
tickers = pd.read_csv('data/symbols_sp500.csv')['Symbol'].tolist()  #[:20]

# Step 1: Get data
print("Fetching data...")
raw_data = fetch_all_tickers(tickers)  # limit for testing

# Step 2: Add indicators
print("Calculating indicators...")
data_with_indicators = add_indicators(raw_data)

# Step 3: Prepare dataset & train model
print("Preparing dataset...")
#dataset = prepare_dataset(data_with_indicators)
dataset = prepare_dataset(data_with_indicators, threshold=threshold, future_days=future_days)

print("Dataset columns:", dataset.columns.tolist())
print(dataset.head())

print("Training model...")
train_and_save_model(dataset)

''' Step 4: Load model and make predictions 
print("Predicting...")
model = load_model("models/xgb_model.pkl")

# Filter: RSI between 30 and 70, prediction prob > 0.7
preds_df, chart_data, chart_figs = predict_all(
    model,
    data_with_indicators,
    rsi_range=(30, 70),
    min_prob=0.7,
    generate_charts=True
)

print(preds_df.head())
'''


'''
model = load_model("models/xgb_model.pkl")
predictions = predict_all(model, data_with_indicators)

# Sort and display top predictions
top_preds = predictions.sort_values(by='Probability', ascending=False)
print(top_preds.head(10))
'''