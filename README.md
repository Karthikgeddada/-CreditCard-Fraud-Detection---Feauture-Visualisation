# -CreditCard-Fraud-Detection---Feauture-Visualisation
This project implements a Credit Card Fraud Detection model using K-Nearest Neighbors (KNN). The dataset used is a subset of credit card transactions, where the goal is to classify transactions as fraudulent (1) or normal (0) based on selected features.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


df_new = pd.read_csv("/content/creditcard.csv")
df_new.head()

X = df_new[['V1', 'Amount']].values
y = df_new['Class'].values



plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Normal', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Fraud', alpha=0.5)
plt.xlabel("V1 (Proxy for Distance from Home)")
plt.ylabel("Amount (Proxy for Ratio to Median Purchase Price)")
plt.legend()
plt.title("Credit Card Fraud Detection - Feature Visualization")
plt.show()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)



y_pred = knn.predict(X_test_scaled)



accuracy = accuracy_score(y_test, y_pred)
accuracy
