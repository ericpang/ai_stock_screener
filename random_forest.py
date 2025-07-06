from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

def prepare_dataset(data_dict):
    rows = []
    for ticker, df in data_dict.items():
        df = df.copy().dropna()

        # Calculate future return (30-day forward % change)
        df['FutureReturn'] = df['Close'].shift(-30) / df['Close'] - 1

        # Create binary label: 1 if future return >= 10%
        df['Label'] = (df['FutureReturn'] >= 0.10).astype(int)

        for i in range(len(df) - 30):
            row = {
                'Ticker': ticker,
                'RSI': df.iloc[i]['RSI'],
                'SMA50': df.iloc[i]['SMA50'],
                'SMA200': df.iloc[i]['SMA200'],
                'Volume_SMA20': df.iloc[i]['Volume_SMA20'],
                'PE': df.iloc[i]['PE'],
                'Label': df.iloc[i]['Label']
            }
            rows.append(row)

    dataset = pd.DataFrame(rows).dropna()
    return dataset

# Prepare the dataset
dataset = prepare_dataset(stock_data_with_indicators)

# Handle missing or infinite values
dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()

# Split into X and y
X = dataset[['RSI', 'SMA50', 'SMA200', 'Volume_SMA20', 'PE']]
y = dataset['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
