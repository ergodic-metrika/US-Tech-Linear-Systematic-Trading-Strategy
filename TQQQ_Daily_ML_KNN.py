# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 09:16:50 2023

@author: user
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV as rcv
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from IPython import get_ipython

# Plotting graphs
import matplotlib.pyplot as plt

# Machine learning libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Data fetching
#pip install yfinance
import yfinance as yf

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# Fetch data
TQQQ_data = yf.download('TQQQ', start='2015-1-1', end='2023-12-25', auto_adjust=True)
df = TQQQ_data[['Open', 'High', 'Low', 'Close']]
df

# Predictor variables
df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low
df = df.dropna()
X = df[['Open-Close', 'High-Low']]
X.head()

# Target variable
Y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
# Splitting the dataset
split_percentage = 0.7
split = int(split_percentage*len(df))

X_train = X[:split]
Y_train = Y[:split]

X_test = X[split:]
Y_test = Y[split:]

# Instantiate KNN learning model(k=15)
knn = KNeighborsClassifier(n_neighbors=35)

# fit the model
knn.fit(X_train, Y_train)

# Accuracy Score
accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

print ('Train_data Accuracy: %.2f' %accuracy_train)
print ('Test_data Accuracy: %.2f' %accuracy_test)

# Predicted Signal
df['Predicted_Signal'] = knn.predict(X)

# TQQQ Cumulative Returns
df['TQQQ_data_returns'] = np.log(df['Close']/df['Close'].shift(1))
Cumulative_TQQQ_data_returns = df[split:]['TQQQ_data_returns'].cumsum()*100

# Cumulative Strategy Returns
df['Strategy_returns'] = df['TQQQ_data_returns']* df['Predicted_Signal'].shift(1)
Cumulative_Strategy_returns = df[split:]['Strategy_returns'].cumsum()*100

# Plot the results to visualise the performance

plt.figure(figsize=(10,5))
plt.plot(Cumulative_TQQQ_data_returns, color='r',label = 'TQQQ Returns')
plt.plot(Cumulative_Strategy_returns, color='g', label = 'KNN Algo Returns')
plt.legend()
plt.show()


# Calculate Sharpe ratio
Std = Cumulative_Strategy_returns.std()
Sharpe = (Cumulative_Strategy_returns - Cumulative_TQQQ_data_returns)/Std
Sharpe = Sharpe.mean()
print('Sharpe ratio: %.2f'%Sharpe )

####################Finding optimal k value for KNN classifier######################
from sklearn.model_selection import GridSearchCV

# Create a KNN model
knn = KNeighborsClassifier()

# Create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 200)}

# Use grid search to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv=5)  # cv is the number of cross-validation folds

# Fit model to data
knn_gscv.fit(X_train, Y_train)

# Check top-performing n_neighbors value
best_k = knn_gscv.best_params_['n_neighbors']

# Check mean score for the top-performing value of n_neighbors
best_score = knn_gscv.best_score_

print(f"Best k: {best_k}")
print(f"Best cross-validated score: {best_score:.2f}")

####################Final K that maximizes the KNN Algo Returns############
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def evaluate_knn_strategy(k_values, X_train, Y_train, X_test, Y_test, close_prices):
    results = []

    for k in k_values:
        # Initialize KNN with the current k value
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Fit the model with the training data
        knn.fit(X_train, Y_train)
        
        # Make predictions on the test set
        predicted_signal = knn.predict(X_test)
        
        # Ensure close_prices and predicted_signal have the same length
        if len(predicted_signal) != len(close_prices):
            raise ValueError("Length mismatch between predicted signals and close prices")

        # Calculate the strategy returns based on the predicted signal
        # Check for any inf or NaN values in close_prices
        if np.isinf(close_prices).any() or np.isnan(close_prices).any():
            raise ValueError("Inf or NaN values found in close prices")

        # Ensure no division by zero or inf values
        strategy_returns = np.log(close_prices.shift(-1) / close_prices).replace([np.inf, -np.inf], np.nan)
        strategy_returns *= predicted_signal
        
        # Drop NaN values from the strategy returns
        strategy_returns = strategy_returns.dropna()

        # Calculate the cumulative returns of the strategy
        cumulative_returns = strategy_returns.cumsum().iloc[-1]
        
        # Append the performance of the strategy
        results.append({
            'k': k,
            'Cumulative_Returns': cumulative_returns
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Check if results_df contains only NaN in 'Cumulative_Returns'
    if results_df['Cumulative_Returns'].isna().all():
        raise ValueError("Cumulative_Returns contains only NaN values.")

    return results_df

# ... rest of the code to prepare the data and call evaluate_knn_strategy ...

# Call the function and handle potential errors
try:
    results_df = evaluate_knn_strategy(k_values, X_train, Y_train, X_test, Y_test, close_prices)
    best_k = results_df.loc[results_df['Cumulative_Returns'].idxmax()]
    print(f"Best k: {best_k['k']}")
    print(f"Best Cumulative Returns: {best_k['Cumulative_Returns']}")
except ValueError as e:
    print(f"Error: {e}")