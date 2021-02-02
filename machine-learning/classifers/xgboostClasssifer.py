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

from xgboost import XGBClassifier

xgb_model = XGBClassifier().fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print('Accuracy:', accuracy_score(y_pred, y_test))

# #Tuning
# xgb_grid = {
#     "n_estimators" : [100,500,1000, 2000],
#     "subsample" : [0.6,0.8,1.0],
#     "max_depth" : [3,4,5,6],
#     "learning_rate" : [0.1, 0.01, 0.02, 0.05],
#     "min_samples_split" : [2,5,10]
# }

# from sklearn.model_selection import GridSearchCV

# xgb_cv_model = GridSearchCV(xgb_model, xgb_grid, cv=10, n_jobs=-1, verbose=2)
# xgb_cv_model.fit(X_train,y_train)
# print(xgb_cv_model.best_params_)


xgb_tuned = XGBClassifier(learning_rate=0.01, max_dept=6, min_samples_split=2, n_estimators=100, subsample=0.8).fit(X_train,y_train)
y_tuned_pred = xgb_tuned.predict(X_test)
print('Tuned Error:', np.sqrt(mean_squared_error(y_test, y_tuned_pred)))