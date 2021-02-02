from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, classification_report, roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier


diabetes = pd.read_csv('../../reading_data/diabetes.csv')
df = diabetes.copy()
df = df.dropna()

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30)

#Scale
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Model
from sklearn.neural_network import MLPClassifier

model = MLPClassifier().fit(X_train_scaled, y_train)

# print(model.coefs_)
y_pred = model.predict(X_test_scaled)
print('Accuracy:', accuracy_score(y_test, y_pred))


mlpc_params = {
    'alpha': [0.1, 0.01, 0.02, 0.005, 0.0001, 0.00001],
    'hidden_layer_sizes': [(10,10,10), (100, 100, 100), (100,100), (3,5), (5,3)],
    'activation': ['relu', 'logistic'],
    'solver' : ["lbfgs", "adam", "sgd"]
}

# from sklearn.model_selection import GridSearchCV

# mlp_cv_model = GridSearchCV(model, mlpc_params, cv = 10, verbose=2, n_jobs=-1)
# mlp_cv_model.fit(X_train_scaled, y_train)

# print(mlp_cv_model.best_params_)

tuned_model = MLPClassifier(activation='relu', alpha=0.1, hidden_layer_sizes=(100,100), solver='sgd').fit(X_train_scaled, y_train)
y_tuned_pred = tuned_model.predict(X_test_scaled)
print('Accuracy:', accuracy_score(y_test, y_tuned_pred))

