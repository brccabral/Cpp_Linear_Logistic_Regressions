# %%
# import the necessary libraries
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn import metrics
# import seaborn as sns
from sklearn import preprocessing

# %%
# read and print the data
data = pd.read_csv("../dataset/winedata-header.csv", sep=",")
data.head()

# %%
# Separate features from output
X = data.drop(["quality"], axis=1)
y = data["quality"]

# %%
# Train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# %%
# Normalize data
scaler = preprocessing.StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.fit_transform(X_test)

# %%
# Fit data to model
regressor = LinearRegression(fit_intercept=True)
regressor.fit(X_train_norm, y_train)
# %%
# Predict on training data - Python
y_hat_sk = regressor.predict(X_train_norm)

# %%
# Predict on training data - scratch model
y_train_hat = pd.read_csv("../dataset/y_train_hat.txt", header=None)

# %%
# Cost function
cost = pd.read_csv("../dataset/cost.txt", header=None)

# %%
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlabel("Epoch")
ax.set_ylabel(r"Cost $J(\theta)$")
ax.plot(cost)
ax.set_title(
    "Training Cost vs. Training Epoch for Multivariate Wine Quality Regression"
)
plt.show()

# %%
# Comparison Scratch vs. Python
plt.figure(figsize=(12, 8))
plt.scatter(y_train, y_hat_sk, c="b", label="SciKit-Learn", s=50)
plt.scatter(y_train, y_train_hat, c="r", label="Custom Function")
plt.ylabel("Predicted Quality $\hat{y_i}$")
plt.xlabel("Quality $y_i$")
plt.legend(loc=2)
plt.grid()
plt.show()

# %%
