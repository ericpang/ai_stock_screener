# model_train.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from config import TARGET_RETURN, PREDICTION_DAYS

from tqdm import tqdm

def prepare_dataset(data_dict, threshold=0.10, future_days=30):
    rows = []

    for ticker, df in data_dict.items():
        df = df.dropna(subset=['RSI', 'SMA50', 'SMA200', 'Volume_SMA20']).copy()
        if len(df) < future_days:
            continue

        df['FutureReturn'] = df['Close'].shift(-future_days) / df['Close'] - 1
        df['Label'] = (df['FutureReturn'] >= threshold).astype(int)

        for i in range(len(df)): # - future_days):
            row = {
                'Ticker': ticker,
                'Date': df.index[i],  # ⬅️ this assumes 'Date' is the index of the original df
                'RSI': df.iloc[i]['RSI'],
                'SMA50': df.iloc[i]['SMA50'],
                'SMA200': df.iloc[i]['SMA200'],
                'Volume_SMA20': df.iloc[i]['Volume_SMA20'],
                'Label': df.iloc[i]['Label'],
                'Close': df.iloc[i]['Close']
            }
            rows.append(row)

    return pd.DataFrame(rows).dropna()





def train_and_save_model(dataset, output_path='models/xgb_model.pkl'):

    #print("Dataset columns:", dataset.columns)
    #print("Dataset preview:\n", dataset.head())
    #print("Dataset length:", len(dataset))

    X = dataset[['RSI', 'SMA50', 'SMA200', 'Volume_SMA20']]
    y = dataset['Label']
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    label_counts = y.value_counts()
    print("Label distribution:\n", label_counts)

    if len(label_counts) < 2 or label_counts.min() < 2:
        print("❗ Not enough samples in each class for stratified split.")
        return  # or raise error, or skip training

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))
    pickle.dump(model, open(output_path, 'wb'))
