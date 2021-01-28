from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, classification_report, roc_auc_score,roc_curve
from sklearn.svm import SVC

diabetes = pd.read_csv('../../reading_data/diabetes.csv')
df = diabetes.copy()
df = df.dropna()

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30)

svm_model = SVC(kernel="rbf").fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print('Accuracy:', accuracy_score(y_pred, y_test))
print('Rapor:', classification_report(y_pred, y_test))

params = {
    "C": [0.0001, 0.001, 0.1,1,5,10,100],
    "gamma" : [0.0001, 0.001, 0.1,1,5,10,100]
}

# from sklearn.model_selection import GridSearchCV

# svc_cv = GridSearchCV(svm_model, params, cv=10, verbose=2, n_jobs=-1)
# svc_cv.fit(X_train, y_train)
# print(svc_cv.best_params_)

svm_tuned_model = SVC(kernel="linear", C=9, gamma=0.0001).fit(X_train, y_train)
y_tuned_pred = svm_tuned_model.predict(X_test)

print('Tuned Accuracy:', accuracy_score(y_tuned_pred, y_test))
print('Tuned Rapor:', classification_report(y_tuned_pred, y_test))

