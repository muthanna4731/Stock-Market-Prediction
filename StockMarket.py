import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

#Load the data
data = pd.read_csv('/Users/muthanna/Documents/coding/python/StockMarketPrediction/monthly_IBM.csv')
df = pd.DataFrame(data)

#PREPROCESSING
df['Date'] = pd.to_datetime(df['timestamp'])
df = df.drop(['timestamp'], axis=1) #axis=1 means column
df.set_index('Date', inplace=True) #Sets index as date//inplane=True means changes are updated to original df
df = df.sort_index()
df = df.dropna() #drops rows with missing values
print(df.head())

print()

#Visualizing the Close Price
df['close'].plot(figsize=(12, 6), title='Close Price Stock Trend', label='Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend() #Shows the line label
plt.show()

#Feature Engineering
df['Range'] = df['high'] - df['low'] #Range of the stock in a month
df['Percentage change'] = (df['close']-df['open'])/df['open']*100 #Percentage change in stock price

#Splitting the data
X = df.drop('close', axis=1)
y = df['close']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #20% of data is used for testing//Random=0 allows us to access the same shuffled set of data
split = int(0.8*len(df))
X_train,X_test,y_train,y_test = X[:split],X[split:],y[:split],y[split:]

#Model
model = LinearRegression()

# Apply RFE//Feature Selection
rfe = RFE(estimator=model, n_features_to_select=3)
rfe.fit(X_train, y_train)

# Selected features
selected_features = X_train.columns[rfe.support_]
print("Selected Features:", selected_features)
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

print()

model.fit(X_train_selected,y_train)
predict = model.predict(X_test_selected)

# Evaluate the model
mse = mean_squared_error(y_test, predict)
r2 = r2_score(y_test, predict)

print(f"MSE: {mse}")
print(f"R^2: {r2}")

y_test_sorted = y_test.sort_index()
predict_sorted = pd.Series(predict, index=y_test_sorted.index)
#Plotting the prediction
plt.figure(figsize=(12, 6))
y_test_sorted.plot(label='Close Price')
predict_sorted.plot(label='Predicted Close Price')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()