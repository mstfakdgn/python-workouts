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

from sklearn.ensemble import RandomForestClassifier

rfm = RandomForestClassifier().fit(X_train, y_train)
y_pred = rfm.predict(X_test)
print('Accuracy:' , accuracy_score(y_pred, y_test))

# rf_params = {
#     'max_depth': [2,5,8,10],
#     'max_features': [2,5,8],
#     'n_estimators': [10,500,1000],
#     'min_samples_split' : [2,5,10]
# }

# from sklearn.model_selection import GridSearchCV

# # n_jobs = -1 parameter is to much full performance
# rf_cv_model = GridSearchCV(rfm, rf_params, cv=10, verbose=2, n_jobs=-1)
# rf_cv_model.fit(X_train, y_train)
# print(rf_cv_model.best_params_)

rf_tuned = RandomForestClassifier(max_depth=5, max_features=8, n_estimators=10, min_samples_split=2)
rf_tuned.fit(X_train, y_train)
y_tuned_pred = rf_tuned.predict(X_test)
print('Tuned Score:', accuracy_score(y_test, y_tuned_pred))

Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                          index=X_train.columns)

Importance.sort_values(by = "Importance", axis = 0, ascending= True).plot(kind = "barh", color = "r")
plt.xlabel("Değişken Önem düzeyleri")
plt.show()