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

from lightgbm import LGBMRegressor

lgbm = LGBMRegressor()
lgbm_model = lgbm.fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test, num_iteration = lgbm_model.best_iteration_)
print('Error:', np.sqrt(mean_squared_error(y_pred, y_test)))

lgbm_params = {
    "colsample_bytree": [0.4,0.5,0.6,0.9,1],
    "learning_rate" : [0.01,0.1,0.5,1],
    "n_estimators":[20,40,100,200, 500, 1000],
    "max_depth": [1,2,3,4,5,6,7,8]
}

# from sklearn.model_selection import GridSearchCV

# lgbm_cv_model = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=2)
# lgbm_cv_model.fit(X_train,y_train)
# print(lgbm_cv_model.best_params_)

lgbm_tuned_model = LGBMRegressor(learning_rate=0.1, max_depth=6, n_estimators=20, colsample_bytree=0.5).fit(X_train, y_train)
y_tuned_pred = lgbm_tuned_model.predict(X_test)
print('Tuned Error:', np.sqrt(mean_squared_error(y_test, y_tuned_pred)))