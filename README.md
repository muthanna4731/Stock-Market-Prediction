# Stock Market Prediction using Facebook Prophet

## Overview  
This project is a **machine learning-based stock market prediction system** that forecasts **monthly closing prices** for stocks using **Facebook Prophet**. It integrates **market trends, momentum, and volatility indicators** to enhance prediction accuracy.  

 **Key Features:**  
- Data preprocessing and feature engineering  
- Time-series forecasting with **Facebook Prophet**  
- **S&P 500 data** as a market trend feature  
- **Interactive Streamlit dashboard** for easy predictions  
- Model evaluation using **MAE, MSE, and RMSE**


## Features  

### **Data Preprocessing**  
- Converted timestamps to `datetime` format for time-series analysis.  
- Removed missing values and handled inconsistencies.  
- Merged **S&P 500 historical data** as an external market trend feature.  

### **Feature Engineering**  
The model uses the following **financial indicators**:  
- **3-month & 6-month momentum** – Captures short-term and mid-term trends.  
- **12-month volatility** – Measures price fluctuations for stability.  
- **Market trend (S&P 500 data)** – Represents broader market influence.  
- **SMA-10 (Simple Moving Average)** – Identifies short-term price trends.  
- **Percentage Change** – Measures stock price fluctuations.  

### **Forecasting with Facebook Prophet**  
- Adjusted **changepoint_prior_scale** to improve trend adaptability.  
- Configured **monthly and quarterly seasonality** for better long-term predictions.  
- Used external **regressors (market trend, momentum, and volatility)** for enhanced accuracy.  
- **Interactive slider** allows users to **adjust forecast timeline** (up to 36 months).  

### **Model Evaluation**  
- **Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE)** for performance assessment.  
- Compared **actual vs. predicted** prices to verify accuracy.  

### **Interactive Dashboard (Streamlit)**  
- **Upload custom stock data** or use the **default IBM dataset**.  
- View **historical stock trends** and **forecasted prices**  
- Visualize **actual vs. predicted** values using **interactive charts**.  
- **Model performance metrics** displayed on the dashboard.  


## Dataset  
The model works with historical monthly stock data, containing:  
- **Date (timestamp)** – Time reference for each data point.  
- **Open, High, Low, Close** – Essential stock price values.  
- **Volume** – Represents market activity.  

_Default dataset: IBM stock historical prices (monthly intervals)._  
_Users can upload their own CSV file containing stock data._  


## Run Streamlit App
streamlit run your_script.py
