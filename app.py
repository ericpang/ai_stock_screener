# app.py

import streamlit as st
import pandas as pd
from model_predict import load_model, predict_all

st.set_page_config(page_title="AI Stock Screener", layout="wide")

# --- Sidebar filters ---
st.sidebar.header("📊 Filter Predictions")

rsi_min = st.sidebar.slider("Min RSI", min_value=0, max_value=100, value=30)
rsi_max = st.sidebar.slider("Max RSI", min_value=0, max_value=100, value=70)
min_prob = st.sidebar.slider("Min Probability", min_value=0.0, max_value=1.0, value=0.7)

st.sidebar.markdown("⚙️ Label Settings")
future_days = st.sidebar.slider("📅 Days to look ahead", min_value=5, max_value=90, value=30)
return_threshold = st.sidebar.slider("📈 Return threshold (%)", min_value=1, max_value=30, value=10) / 100



# --- Main title ---
st.title("🚀 AI Stock Screener with XGBoost")
st.caption("Predicts high-growth stocks using technical indicators and live data.")

# --- Load model ---
model = load_model("models/xgb_model.pkl")

# --- Load processed data ---
@st.cache_data(show_spinner="🔄 Recalculating dataset...")
def load_processed_data(days, threshold):
    from indicators import add_indicators
    from data_fetch import fetch_all_tickers
    from model_train import prepare_dataset
    import pandas as pd

    tickers = pd.read_csv("data/symbols_sp500.csv")["Symbol"].tolist()  #[:20] # Limit for speed
    data_raw = fetch_all_tickers(tickers)  # Limit for speed
    data_proc = add_indicators(data_raw)

    return prepare_dataset(data_proc, threshold=threshold, future_days=days)

with st.spinner("Fetching data and running predictions..."):
    processed_df = load_processed_data(future_days, return_threshold)

    # Split the dataset by ticker into a dict of DataFrames
    processed_data_dict = {
        ticker: processed_df[processed_df["Ticker"] == ticker].copy()
        for ticker in processed_df["Ticker"].unique()
    }

    preds_df, valid_dfs, charts = predict_all(
        model,
        processed_data_dict,
        rsi_range=(rsi_min, rsi_max),
        min_prob=min_prob,
        generate_charts=True
    )


# --- Display predictions ---
st.subheader("📈 Top Predictions")
st.dataframe(preds_df.reset_index(drop=True), use_container_width=True)

# --- Chart viewer ---
if not preds_df.empty:
    #st.dataframe(preds_df)
    selected = st.selectbox("📉 View chart for a stock", preds_df["Ticker"].tolist())
    if selected in charts:
        st.pyplot(charts[selected])
else:
    st.warning("No stocks met your filter criteria. Try adjusting RSI or Probability thresholds.")

# --- Download CSV ---
if not preds_df.empty:
    st.download_button(
        label="📥 Download Results as CSV",
        data=preds_df.to_csv(index=False).encode(),
        file_name="ai_stock_predictions.csv",
        mime="text/csv"
    )
