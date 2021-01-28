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


import xgboost as xgb

DM_train = xgb.DMatrix(data = X_train, label= y_train)
DM_test = xgb.DMatrix(data = X_test, label= y_test)

from xgboost import XGBRegressor

xgb = XGBRegressor().fit(X_train, y_train)

y_pred = xgb.predict(X_test)
print('Error:', np.sqrt(mean_squared_error(y_pred, y_test)))

# Importance = pd.DataFrame({"Importance": xgb.feature_importances_*100},
#                           index=X_train.columns)

# Importance.sort_values(by = "Importance", axis = 0, ascending= True).plot(kind = "barh", color = "r")
# plt.xlabel("Değişken Önem düzeyleri")
# plt.show()



#Tuning
xgb_grid = {
    "colsample_bytree" : [0.4,0.5,0.6,0.9,1],
    "n_estimators" : [100,200,500,1000],
    "max_depth" : [2,3,4,5,6],
    "learning_rate" : [0.1,0.01,0.5]
}

# from sklearn.model_selection import GridSearchCV

# xgb_cv_model = GridSearchCV(xgb, xgb_grid, cv=10, n_jobs=-1, verbose=2)
# xgb_cv_model.fit(X_train,y_train)
# print(xgb_cv_model.best_params_)

xgb_tuned = XGBRegressor(colsample_bytree=0.6, learning_rate=0.1, max_depth=2, n_estimators=1000).fit(X_train,y_train)
y_tuned_pred = xgb_tuned.predict(X_test)
print('Tuned Error:', np.sqrt(mean_squared_error(y_test, y_tuned_pred)))
