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

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier().fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_pred, y_test))

gbm_params = {
    "learning_rate" : [0.001,0.01,0.1,0.05],
    "n_estimators":[100,500,1000],
    "max_depth": [3,5,10],
    "min_samples_split" : [2,5,10]
}

# from sklearn.model_selection import GridSearchCV

# gbm_cv_model = GridSearchCV(model, gbm_params, cv=10, n_jobs=-1, verbose=2)
# gbm_cv_model.fit(X_train,y_train)
# print(gbm_cv_model.best_params_)

model_tuned = GradientBoostingClassifier(learning_rate=0.01, max_depth=10, min_samples_split=10, n_estimators=500).fit(X_train, y_train)
y_tuned_pred = model_tuned.predict(X_test)
print('Accuracy:', accuracy_score(y_tuned_pred, y_test))