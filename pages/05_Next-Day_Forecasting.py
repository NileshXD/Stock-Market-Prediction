# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import random as rn
# import yfinance as yf
# import streamlit as st
# import datetime as dt
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from sklearn.preprocessing import MinMaxScaler
# np.random.seed(1)
# tf.random.set_seed(1)
# rn.seed(1)
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM
# from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
# from millify import millify


# st.subheader('Next-Day Forecasting with Long-Short Term Memory (LSTM)')

# csv = pd.read_csv('symbols.csv')
# symbol = csv['Symbol'].tolist()
# for i in range(0, len(symbol)):
#     symbol[i] = symbol[i] + ".NS"

# # creating sidebar
# ticker = st.selectbox(
#         'Enter or Choose NSE listed Stock Symbol',
#         symbol, index=symbol.index('TRIDENT.NS'))

# def my_LSTM(ticker):
#         try:
#             start = dt.datetime.today() - dt.timedelta(5*365)
#             end = dt.datetime.today()

#             df = yf.download(ticker,start,end)
#             df =  df.reset_index()
#             df['Date'] = pd.to_datetime(df['Date']).dt.date
#             st.write('It will take some seconds to fit the model....')
#             data = df.sort_index(ascending=True, axis=0)
#             new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
#             for i in range(0, len(data)):
#                 new_data['Date'][i] = data['Date'][i]
#                 new_data['Close'][i] = data['Close'][i]

#         # setting index
#             new_data.index = new_data.Date
#             new_data.drop('Date', axis=1, inplace=True)

#         # creating train and test sets
#             dataset = new_data.values

#             train = dataset[0:987, :]
#             valid = dataset[987:, :]

#         # converting dataset into x_train and y_train
#             scaler = MinMaxScaler(feature_range=(0, 1))
#             scaled_data = scaler.fit_transform(dataset)

#             x_train, y_train = [], []
#             for i in range(60, len(train)):
#                 x_train.append(scaled_data[i - 60:i, 0])
#                 y_train.append(scaled_data[i, 0])
#             x_train, y_train = np.array(x_train), np.array(y_train)

#             x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#         # create and fit the LSTM network
#             model = Sequential()
#             model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
#             model.add(LSTM(units=50))
#             model.add(Dense(1))

#             model.compile(loss='mean_squared_error', optimizer='adam')
#             model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
#             st.success('Model Fitted')
#         # predicting 246 values, using past 60 from the train data
#             inputs = new_data[len(new_data) - len(valid) - 60:].values
#             inputs = inputs.reshape(-1, 1)
#             inputs = scaler.transform(inputs)

#             X_test = []
#             for i in range(60, inputs.shape[0]):
#                 X_test.append(inputs[i - 60:i, 0])
#             X_test = np.array(X_test)

#             X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#             closing_price = model.predict(X_test)
#             closing_price = scaler.inverse_transform(closing_price)


#         # for plotting
#             train = data[:987]
#             valid = data[987:]
#             valid['Predictions'] = closing_price

#             st.write('#### Actual VS Predicted Prices')

#             fig_preds = go.Figure()
#             fig_preds.add_trace(
#             go.Scatter(
#                 x=train['Date'],
#                 y=train['Adj Close'],
#                 name='Training data Closing price'
#             )
#             )

#             fig_preds.add_trace(
#             go.Scatter(
#                 x=valid['Date'],
#                 y=valid['Adj Close'],
#                 name='Validation data Closing price'
#             )
#             )

#             fig_preds.add_trace(
#             go.Scatter(
#                 x=valid['Date'],
#                 y=valid['Predictions'],
#                 name='Predicted Closing price'
#             )
#             )

#             fig_preds.update_layout(legend=dict(
#             orientation='h',
#             yanchor='bottom',
#             y=1,
#             xanchor='left',
#             x=0)
#             , height=600, title_text='Predictions on Validation Data', template='gridon'
#             )

#             st.plotly_chart(fig_preds, use_container_width=True)

#         # metrics
#             mae = mean_absolute_error(closing_price, valid['Adj Close'])
#             rmse = np.sqrt(mean_squared_error(closing_price, valid['Adj Close']))
#             accuracy = r2_score(closing_price, valid['Adj Close'])*100

#             with st.container():
#                 st.write('#### Metrics')
#                 col_11, col_22, col_33 = st.columns(3)
#                 col_11.metric('Absolute error between predicted and actual value', round(mae,2))
#                 col_22.metric('Root mean squared error between predicted and actual value', round(rmse,2))

#         # forecasting
#             real_data = [inputs[len(inputs) - 60:len(inputs + 1), 0]]
#             real_data = np.array(real_data)
#             real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

#             prediction = model.predict(real_data)
#             prediction = scaler.inverse_transform(prediction)
#             st.write('#### Next-Day Forecasting')

#             with st.container():
#                 col_111, col_222, col_333 = st.columns(3)
#                 col_111.metric(f'Closing Price Prediction of the next trading day for {ticker} is',f' ₹ {str(round(float(prediction),2))}')

#         except:
#             st.warning("Oops! you can't go ahead!!")
#             st.warning("The company you selected is listed newly...so we can't gather data.")

# my_LSTM(ticker)







import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from millify import millify

st.subheader('Next-Day Forecasting using Machine Learning (Random Forest & Linear Regression)')

# Load symbols
csv = pd.read_csv('symbols.csv')
symbol = csv['Symbol'].tolist()
symbol = [s + ".NS" for s in symbol]

# Sidebar
ticker = st.selectbox(
    'Enter or Choose NSE listed Stock Symbol',
    symbol, 
    index=symbol.index('TRIDENT.NS')
)

def forecast_next_day_sklearn(ticker):

    try:
        # -----------------------------
        # 1️⃣ LOAD DATA
        # -----------------------------
        start = dt.datetime.today() - dt.timedelta(5*365)
        end = dt.datetime.today()

        df = yf.download(ticker, start, end)
        df = df.reset_index()

        if "Close" not in df.columns:
            st.error("Yahoo Finance did not return 'Close' field.")
            return

        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Close']]

        # -----------------------------
        # 2️⃣ FEATURE ENGINEERING
        # -----------------------------
        df['Prev_Close'] = df['Close'].shift(1)
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['Return'] = df['Close'].pct_change()

        df = df.dropna()

        # -----------------------------
        # 3️⃣ TRAIN / TEST SPLIT
        # -----------------------------
        X = df[['Prev_Close', 'MA_5', 'MA_10', 'Return']]
        y = df['Close']

        split = int(len(df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # -----------------------------
        # 4️⃣ TRAIN MODELS
        # -----------------------------
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)

        # -----------------------------
        # 5️⃣ VALIDATION PREDICTIONS
        # -----------------------------
        lr_pred = lr.predict(X_test)
        rf_pred = rf.predict(X_test)

        # -----------------------------
        # 6️⃣ ENSEMBLE PREDICTION = average of both models
        # -----------------------------
        ensemble_pred = (lr_pred + rf_pred) / 2

        df_test = df.iloc[split:].copy()
        df_test['Prediction'] = ensemble_pred

        # -----------------------------
        # 7️⃣ PLOT — Actual vs Predicted
        # -----------------------------
        st.write("### Actual vs Predicted Closing Prices")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_test['Date'], y=df_test['Close'],
                                 mode='lines', name='Actual Close'))
        fig.add_trace(go.Scatter(x=df_test['Date'], y=df_test['Prediction'],
                                 mode='lines', name='Predicted Close'))

        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # 8️⃣ METRICS
        # -----------------------------
        mae = mean_absolute_error(y_test, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        r2 = r2_score(y_test, ensemble_pred) * 100

        st.write("### Model Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", round(mae, 2))
        c2.metric("RMSE", round(rmse, 2))
        c3.metric("R² Score (%)", round(r2, 2))

        # -----------------------------
        # 9️⃣ NEXT DAY FORECASTING
        # -----------------------------
        last_row = df.iloc[-1:]
        next_X = last_row[['Prev_Close', 'MA_5', 'MA_10', 'Return']]

        pred_lr = lr.predict(next_X)[0]
        pred_rf = rf.predict(next_X)[0]

        next_day_prediction = (pred_lr + pred_rf) / 2  # ensemble

        st.write("### Next Trading Day Forecast")
        st.metric(f"Predicted next-day closing price for {ticker}",
                  f"₹ {round(next_day_prediction, 2)}")

    except Exception as e:
        st.warning("Something went wrong in forecasting.")
        st.error(str(e))


forecast_next_day_sklearn(ticker)

