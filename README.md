# Stock Market Prediction using Linear Regression

## Overview
This project is a simple implementation of stock market prediction using **Linear Regression**. The model predicts the monthly closing price of IBM stock based on historical stock data. It includes essential preprocessing, feature engineering, and model evaluation steps. It is a starting point for time-series prediction tasks.

## Features
- **Data Preprocessing:**
  - Converted timestamps to `datetime` format for better indexing.
  - Removed missing values to ensure data integrity.
  - Sorted the dataset by date for chronological consistency.

- **Feature Engineering:**
  - Added new features such as:
    - `Range`: Difference between the high and low prices of the stock.
    - `Percentage Change`: Percent change between the opening and closing prices.

- **Visualization:**
  - Plotted the stock's closing price trend over time for data analysis.

- **Machine Learning:**
  - Selected important features using **Recursive Feature Elimination (RFE)**.
  - Built and trained a **Linear Regression Model**.
  - Evaluated the model using metrics such as:
    - Mean Squared Error (MSE)
    - R² Score

## Dataset
The dataset (`monthly_IBM.csv`) contains historical monthly stock prices for IBM, including columns such as:
- `timestamp`: Date of the stock data.
- `open`: Opening price of the stock.
- `high`: Highest price of the stock.
- `low`: Lowest price of the stock.
- `close`: Closing price of the stock.

## Dependencies
This project uses the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `sklearn`
