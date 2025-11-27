import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px


# ---------------------------------------------------------
# 1. SAFE DATA LOADER
# ---------------------------------------------------------
def load_price_data(ticker: str) -> pd.DataFrame:
    """Download price data reliably with fallback handling."""
    start = dt.datetime.today() - dt.timedelta(days=5 * 365)
    end = dt.datetime.today()

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column"
    )

    # Handle empty or None
    if df is None or df.empty:
        st.error("❌ No data returned for this ticker. Try another one.")
        st.stop()

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # Ensure Close exists
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            st.error("❌ Price data does not contain Close/Adj Close.")
            st.stop()

    # Create Adj Close if missing
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    # Ensure OHLV exist
    for col in ["Open", "High", "Low", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df.reset_index()
    return df


# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Daily returns
    df["Return_1d"] = df["Close"].pct_change()

    # Technical indicators
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # MACD
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Volatility
    df["Volatility"] = df["Return_1d"].rolling(10).std()

    # Target: next-day close
    df["Target"] = df["Close"].shift(-1)

    return df.dropna()


# ---------------------------------------------------------
# 3. CHECK IF ENOUGH DATA EXISTS
# ---------------------------------------------------------
def check_data(df, min_rows=150):
    if df is None or df.empty or len(df) < min_rows:
        st.error(
            f"❌ Not enough data to train the model.\n"
            f"Required: {min_rows} rows\nFound: {len(df)} rows"
        )
        st.stop()


# ---------------------------------------------------------
# 4. TRAIN MODELS
# ---------------------------------------------------------
def train_models(df):
    X = df.drop(["Target", "Date"], axis=1, errors="ignore")
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200, random_state=42)

    lr.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)

    # Predictions
    lr_pred = lr.predict(X_test_scaled)
    rf_pred = rf.predict(X_test_scaled)

    return lr, rf, scaler, X_test, y_test, lr_pred, rf_pred


# ---------------------------------------------------------
# 5. STREAMLIT UI
# ---------------------------------------------------------
st.title("📈 Next-Day Stock Price Forecasting (ML Version)")

ticker = st.text_input("Enter Stock Ticker (ex: AAPL, RELIANCE.NS):", "AAPL")

if st.button("Predict Next-Day Price"):
    st.info("⏳ Fetching data…")

    df = load_price_data(ticker)
    st.success("✔ Data Loaded")

    st.info("⏳ Building features…")
    feat_df = build_features(df)
    check_data(feat_df)

    st.success(f"✔ {len(feat_df)} rows ready for training")

    st.info("⏳ Training Machine Learning models…")
    lr, rf, scaler, X_test, y_test, lr_pred, rf_pred = train_models(feat_df)
    st.success("✔ Models Trained")

    # -----------------------------------------------------
    # DISPLAY METRICS
    # -----------------------------------------------------
    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)
    col1.metric("Linear Regression RMSE", f"{np.sqrt(mean_squared_error(y_test, lr_pred)):.4f}")
    col2.metric("Random Forest RMSE", f"{np.sqrt(mean_squared_error(y_test, rf_pred)):.4f}")

    # -----------------------------------------------------
    # NEXT-DAY PREDICTION
    # -----------------------------------------------------
    last_row = feat_df.drop(["Target", "Date"], axis=1).iloc[-1]
    last_scaled = scaler.transform([last_row])

    lr_next = lr.predict(last_scaled)[0]
    rf_next = rf.predict(last_scaled)[0]

    st.subheader("📌 Predicted Next-Day Closing Price")
    st.metric("Random Forest Prediction", f"${rf_next:.2f}")
    st.caption("Random Forest performs better than Linear Regression for non-linear markets.")

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------
    st.subheader("📊 Feature Importance (Random Forest)")
    fi = pd.DataFrame({
        "Feature": feat_df.drop(["Target", "Date"], axis=1).columns,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.bar_chart(fi.set_index("Feature"))

    # -----------------------------------------------------
    # FORECAST VISUALIZATION
    # -----------------------------------------------------
    st.subheader("📈 Forecast vs Actual")

    chart_df = pd.DataFrame({
        "Actual": y_test.values,
        "RF_Pred": rf_pred,
        "LR_Pred": lr_pred
    })

    st.line_chart(chart_df)

    st.success("🎉 Forecasting completed successfully!")
