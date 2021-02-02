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

from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier().fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)
print('Accuracy:', accuracy_score(y_pred, y_test))


lgbm_params = {
    "n_estimators": [100,500,1000,2000],
    "subsample" : [0.6,0.8,1.0],
    "max_depth": [3,4,5,6],
    "learning_rate":[0.1,0.01,0.02,0.05],
    'min_child_samples' :[5,10,20]
}

# from sklearn.model_selection import GridSearchCV

# lgbm_cv_model = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=2)
# lgbm_cv_model.fit(X_train,y_train)
# print(lgbm_cv_model.best_params_)

lgbm_tuned_model = LGBMClassifier(learning_rate=0.02, max_depth=5, num_leaves=4, min_child_samples=5, n_estimators=100, subsample=0.6).fit(X_train, y_train)
y_pred = lgbm_tuned_model.predict(X_test)
print('Accuracy:', accuracy_score(y_pred, y_test))