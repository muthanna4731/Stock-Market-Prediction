import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st


# -------------------------------- Load & Preprocess User Data --------------------------------

# Streamlit App
st.title("💰 Stock Market Prediction System\n***A Fun Way to Predict Your :rainbow[Stock Prices]***")
st.write("Upload a stock dataset and get predictions using the Prophet Model.")

data_source = st.radio("Choose Data Source:", ("Upload CSV", "Use Default (IBM Stock)"))

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file (***Must** contain 'timestamp' & 'close'*)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("👍 File uploaded successfully!")
    else:
        st.warning("✋ Stop right there! Upload CSV file to Proceed")
        st.stop()  # Stop execution until a file is uploaded

else:
    df = pd.read_csv('/Users/muthanna/Documents/coding/python/Stock-Market-Prediction/monthly_IBM.csv')
    st.info("📈 Using the default IBM stock dataset.")

# Date in datetime format
if "timestamp" in df.columns:
    df["ds"] = pd.to_datetime(df["timestamp"])
    df = df.drop(columns=['timestamp']).dropna()
elif "Date" in df.columns:
    df["ds"] = pd.to_datetime(df["Date"])
    df = df.drop(columns=['Date']).dropna()
else:
    st.error("🙅‍♂️ Nope! CSV file must contain either 'timestamp' or 'Date' column!")
    st.stop()

df.columns = df.columns.str.lower()
df = df.rename(columns={'close': 'y'})

# -------------------------------- Market Trend Data (S&P 500) --------------------------------


sp500 = pd.read_csv("/Users/muthanna/Documents/snp500.csv")

sp500["Date"] = pd.to_datetime(sp500["Date"])

sp500 = sp500.reset_index() # Reset index to make 'Date' a column

sp500.rename(columns={'Date': 'ds', 'Close': 'market_trend'}, inplace=True)

sp500 = sp500.set_index('ds')

sp500.columns = [col[0] if isinstance(col, tuple) else col for col in sp500.columns] #Flattening multi-index columns (if they exist)

sp500 = sp500.reindex(df['ds'], method='nearest') #Aligning S&P 500 data with stock data

sp500.loc[sp500.duplicated(subset=['market_trend'], keep='first'), 'market_trend'] = np.nan #Replacing duplicate values with NaN

sp500['market_trend'] = sp500['market_trend'].interpolate(method='linear') #Making market trend data continuous

sp500 = sp500.reset_index()

sp500 = sp500[['ds', 'market_trend']]

df = df.merge(sp500, on="ds", how="left")

df['market_trend'] = df['market_trend'].ffill().bfill()

# -------------------------------- Feature Engineering --------------------------------

#Generating useful features
df['3mo_momentum'] = df['y'].pct_change(3).shift(1) 
df['6mo_momentum'] = df['y'].pct_change(6).shift(1)
df['12mo_volatility'] = df['y'].pct_change().rolling(12).std() # 12mo volatility (more stable)
df['Percentage Change'] = (df['y'] - df['open']) / df['open'] * 100
df['SMA_10'] = df['y'].rolling(window=10).mean()

#Fill NaN values (for initial SMA calculations)
df.fillna(method='bfill', inplace=True)

#Normalize features
features = ['y', 'market_trend', 'Percentage Change', 'SMA_10', '12mo_volatility','3mo_momentum', '6mo_momentum']
for f in features:
    df[f] = (df[f] - df[f].mean()) / df[f].std()


# -------------------------------- Train-Test Split --------------------------------

# Split dataset into 95% training, 5% testing
train_size = int(len(df) * 0.95)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# -------------------------------- Prophet Model --------------------------------

model = Prophet(changepoint_prior_scale=1.0, seasonality_mode='additive')
model.add_seasonality(name='monthly', period=30.5, fourier_order=6)
model.add_seasonality(name='quarterly', period=90, fourier_order=8)
#Add regressors
model.add_regressor('market_trend', standardize=True, mode='additive')
model.add_regressor('Percentage Change', standardize=True, mode='additive')
model.add_regressor('SMA_10', standardize=True, mode='additive')
model.add_regressor('12mo_volatility', standardize=True, mode='multiplicative')
model.add_regressor('3mo_momentum', standardize=True, mode='multiplicative')
model.add_regressor('6mo_momentum', standardize=True, mode='multiplicative')

#Train the model
model.fit(train)

# -------------------------------- Make Future Predictions --------------------------------

future_timeline = st.slider("Set the Forecast Timeline",0,36,12)
future = model.make_future_dataframe(periods = future_timeline, freq='M') #Creating future df
future = pd.concat([test[['ds']], future]).drop_duplicates().reset_index(drop=True) #Ensure the test period is covered

#Merging additional features into the future DataFrame
future = future.merge(df[['ds', 'market_trend', 'Percentage Change', 'SMA_10','12mo_volatility','3mo_momentum','6mo_momentum']], on="ds", how="left")

#Filling missing values using forwarad fill and backward fill
future[['Percentage Change','12mo_volatility','3mo_momentum','6mo_momentum']] = future[['Percentage Change','12mo_volatility','3mo_momentum','6mo_momentum']].ffill().bfill()
future[['market_trend','SMA_10']] = future[['market_trend','SMA_10']].ffill().bfill()

#Filling remaining missing values
future.fillna(method='ffill', inplace=True)  
future.fillna(method='bfill', inplace=True)

print(future.isna().sum()) #Checking if all NaNs are removed

forecast = model.predict(future) #Predict future stock prices

# -------------------------------- Evaluate Model Performance --------------------------------

forecast['yhat'] = forecast['yhat'].rolling(window=5, min_periods=1).mean() #Smooth out the forecasted values
forecast_test = pd.merge(df[['ds', 'y']], forecast[['ds', 'yhat']], on='ds', how='left') #Merge actual values from df with forecast
forecast_test['yhat'] = forecast_test['yhat'].ffill().bfill() #Filling for edge cases
forecast_test = forecast_test.sort_values('ds').reset_index(drop=True)

# Compute evaluation metrics
mae = mean_absolute_error(forecast_test["y"], forecast_test["yhat"])
mse = mean_squared_error(forecast_test["y"], forecast_test["yhat"])
rmse = np.sqrt(mse)

print(f"\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# -------------------------------- Visualize Predictions vs Actual --------------------------------

#Pyplot Graphs
plt.figure(figsize=(12, 6))
plt.plot(forecast_test["ds"], forecast_test["y"], label='Actual', color='black', linewidth=2)
plt.plot(forecast_test["ds"], forecast_test["yhat"], label='Forecast', color='red', linestyle='dashed')
plt.title("Actual vs Forecasted Stock Prices")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

#Streamlit Graphs
forecast['yhat'] = forecast['yhat'].ffill().bfill()

st.markdown("**Stock Forecast**")

forecast_data = pd.DataFrame(
    {
    "Date" : forecast["ds"],
    "Forecast" : forecast["yhat"]
    }
)
st.line_chart(forecast_data,x="Date",y="Forecast")


st.header("**Model Performance**")

# Actual vs Predicted
st.markdown("**Actual vs Predicted Stock Price**")
chart_data = pd.DataFrame({
    "Date": forecast_test["ds"],
    "Actual": forecast_test["y"],
    "Forecast": forecast_test["yhat"]
})

st.line_chart(chart_data, x="Date", y=["Actual", "Forecast"])

col1, col2, col3 = st.columns(3)
col1.metric("MAE",f"{mae:.2f}",border=True)
col2.metric("MSE",f"{mse:.2f}",border=True)
col3.metric("RMSE",f"{rmse:.2f}",border=True)
st.caption(":green[*Lower values indicate higher accuracy*]")
