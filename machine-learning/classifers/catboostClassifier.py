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

from catboost import CatBoostClassifier

model = CatBoostClassifier().fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_pred, y_test))

# catb_params = {
#     "iterations":[200,500,1000,2000],
#     "learning_rate":[0.01,0.03,0.05,0.1],
#     "depth" : [3,4,5,6,7,8]
# }

# from sklearn.model_selection import GridSearchCV

# catb_cv_model = GridSearchCV(model, catb_params, cv=10, n_jobs=-1, verbose=2)
# catb_cv_model.fit(X_train,y_train)
# print(catb_cv_model.best_params_)

catb_tuned_model = CatBoostClassifier(iterations=200,learning_rate=0.03, depth=7).fit(X_train, y_train)
y_tuned_pred = catb_tuned_model.predict(X_test)
print('Tuned Error:', np.sqrt(mean_squared_error(y_test, y_tuned_pred)))