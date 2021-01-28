from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, classification_report, roc_auc_score,roc_curve

diabetes = pd.read_csv('../../reading_data/diabetes.csv')
df = diabetes.copy()
df = df.dropna()

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
print('Accuract Score:', accuracy_score(y_pred, y_test))
print('Cross Val Score:', cross_val_score(nb_model, X_train, y_train, cv=10).mean())