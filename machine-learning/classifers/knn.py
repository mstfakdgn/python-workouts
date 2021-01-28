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

knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

print('Accuracy:', accuracy_score(y_pred, y_test))
print('Rapor:', classification_report(y_pred, y_test))

knn_params = {
    "n_neighbors": np.arange(1, 50)
}

from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_params, cv=10)
knn_cv.fit(X_train, y_train)

print("Best score:" + str(knn_cv.best_score_))
print("Best parameters:" + str(knn_cv.best_params_))


knn = KNeighborsClassifier(n_neighbors=29)
knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
print('Tuned Accuracy:', accuracy_score(y_pred, y_test))
print('Tuned Rapor:', classification_report(y_pred, y_test))