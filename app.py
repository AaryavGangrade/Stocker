import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction with Linear Regression")

# Sidebar for stock selection
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL, MSFT, TSLA)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))

@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, interval='1d')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.dropna(inplace=True)
    return data

data = load_data(ticker, start_date, end_date)

st.subheader(f"{ticker} Close Price History")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data['Close'])
ax.set_xlabel('Date')
ax.set_ylabel('Close Price (USD)')
ax.set_title(f'{ticker} Close Price History')
st.pyplot(fig)

# Feature Engineering
df = data.copy()
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['Volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
df['Min_20'] = df['Close'].rolling(window=20).min()
df['Max_20'] = df['Close'].rolling(window=20).max()
df['Close_lag_1'] = df['Close'].shift(1)
df['Close_lag_2'] = df['Close'].shift(2)
df['Return_lag_1'] = df['Close'].pct_change().shift(1)
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# Train/Test Split
split_ratio = 0.8
split_index = int(len(df) * split_ratio)
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_pred_series = pd.Series(y_pred, index=y_test.index)

# Metrics
model_rmse = np.sqrt(mean_squared_error(y_test, y_pred_series))
model_mae = mean_absolute_error(y_test, y_pred_series)
y_baseline_pred = X_test['Close']
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_baseline_pred))
baseline_mae = mean_absolute_error(y_test, y_baseline_pred)
actual_change = y_test.values - X_test['Close'].values
predicted_change = y_pred_series.values - X_test['Close'].values
model_dir_accuracy = np.mean(np.sign(actual_change) == np.sign(predicted_change)) * 100
up_days = np.mean(actual_change > 0) * 100

# Show metrics
st.subheader("Evaluation Metrics")
st.write(f"**Model RMSE:** {model_rmse:.4f}")
st.write(f"**Baseline RMSE:** {baseline_rmse:.4f}")
st.write(f"**Model MAE:** {model_mae:.4f}")
st.write(f"**Baseline MAE:** {baseline_mae:.4f}")
st.write(f"**Model Directional Accuracy:** {model_dir_accuracy:.2f}%")
st.write(f"**% of 'up' days in test set:** {up_days:.2f}%")

# Actual vs Predicted Plot
st.subheader("Actual vs. Predicted Close Price (Test Set)")
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(y_test.index, y_test, label='Actual Price', color='blue')
ax2.plot(y_test.index, y_pred, label='Predicted Price', color='red', linestyle='--')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price (USD)')
ax2.set_title(f'{ticker} - Actual vs. Predicted Close Price (Test Set)')
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# Show sample data
tab1, tab2 = st.tabs(["Raw Data", "Feature Data"])
with tab1:
    st.dataframe(data.tail(20))
with tab2:
    st.dataframe(df.tail(20))

st.info("Change the ticker in the sidebar to analyze a different stock!")