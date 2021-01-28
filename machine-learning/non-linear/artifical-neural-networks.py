from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

hit = pd.read_csv('../../reading_data/Hitters.csv')
df = hit.copy()
df = df.dropna()
ms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'],
             axis=1).astype('float64', 'int64')
X = pd.concat([X_, ms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
# print(X_train, X_test, y_train, y_test)


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
# print(X_train_scaled)

scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)
# print(X_test_scaled)

# # Model

mlp_model = MLPRegressor().fit(X_train_scaled, y_train)
# Katman sayısı
print(mlp_model.n_layers_)
print(mlp_model.hidden_layer_sizes)


# Predict
y_pred = mlp_model.predict(X_test_scaled)
print(np.sqrt(mean_squared_error(y_test, y_pred)))


# Model Tuning
mlp_params = {
    'alpha': [0.1, 0.01, 0.02, 0.005],
    'hidden_layer_sizes': [(20, 20), (100, 50, 150), (300, 200, 150)],
    'activation': ['relu', 'logistic']
}

from sklearn.model_selection import GridSearchCV

mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv = 10)
mlp_cv_model.fit(X_train_scaled, y_train)

print(mlp_cv_model.best_params_)

mlp_tuned = MLPRegressor(alpha =0.02, hidden_layer_sizes = (100,50,150))
mlp_tuned.fit(X_train_scaled, y_train)
y_tuned_predict = mlp_tuned.predict(X_test_scaled)
print("Tuned". np.sqrt(mean_squared_error(y_test, y_tuned_predict)))