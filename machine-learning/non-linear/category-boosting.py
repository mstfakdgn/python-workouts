import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt 

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

from catboost import CatBoostRegressor

catb = CatBoostRegressor()
catb_model = catb.fit(X_train, y_train)
y_pred = catb_model.predict(X_test)
print('Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

catb_params = {
    "iterations":[200,500,1000,2000],
    "learning_rate":[0.01,0.03,0.05,0.1],
    "depth" : [3,4,5,6,7,8]
}

# from sklearn.model_selection import GridSearchCV

# catb_cv_model = GridSearchCV(catb_model, catb_params, cv=10, n_jobs=-1, verbose=2)
# catb_cv_model.fit(X_train,y_train)
# print(catb_cv_model.best_params_)

catb_tuned_model = CatBoostRegressor(iterations=200,learning_rate=0.1, depth=3).fit(X_train, y_train)
y_tuned_pred = catb_tuned_model.predict(X_test)
print('Tuned Error:', np.sqrt(mean_squared_error(y_test, y_tuned_pred)))