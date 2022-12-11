# %%
# import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# %%
# read and print the data
data = pd.read_csv("../dataset/adult_data.csv", sep=",", header=None)
data.head()

# %%
# Cost function
cost = pd.read_csv("../dataset/cost.txt", header=None)

# %%
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlabel("Epoch")
ax.set_ylabel(r"Cost $J(\theta)$")
ax.plot(cost)
ax.set_title("Training Cost vs. Training Epoch for Logistic Regression")
plt.show()

# %%
# Compare agains Scikit-Learn
X = data.iloc[:, 0:14]
y = data.iloc[:, 14]

mean_data = X.mean()
std_data = X.std()
X = (X - mean_data) / std_data

print("Data shape: " + str(X.shape))
print("Labels shape: " + str(y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Train Data shape: " + str(X_train.shape))
print("Train Labels shape: " + str(y_train.shape))
print("Evaluation Data shape: " + str(X_test.shape))
print("Evaluation Labels shape: " + str(y_test.shape))

# %%
X_train.head()

# %%
logisticRegr = LogisticRegression(solver="lbfgs")
logisticRegr.fit(X_train, y_train.ravel())

# %%
pred_train = logisticRegr.predict(X_train)
print("Train Accuracy: ", 100 - np.mean(np.abs(pred_train - y_train)) * 100)

# %%
pred_test = logisticRegr.predict(X_test)
print("Test Accuracy: ", 100 - np.mean(np.abs(pred_test - y_test)) * 100)

# %%
