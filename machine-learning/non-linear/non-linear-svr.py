import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

hit = pd.read_csv('../../reading_data/Hitters.csv')
df = hit.copy()
df = df.dropna()
ms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis = 1).astype('float64','int64')
X = pd.concat([X_, ms[["League_N", "Division_W", "NewLeague_N"]]], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.25, random_state=42)
# print(X_train, X_test, y_train, y_test)


#Model
from sklearn.svm import SVR

svr_rbf = SVR("rbf").fit(X_train, y_train)

#Predict
y_pred = svr_rbf.predict(X_test)
print("First Error:",np.sqrt(mean_squared_error(y_test, y_pred)))
print("First R2:",r2_score(y_test, y_pred))


#Model Tuning
from sklearn.model_selection import GridSearchCV

svr_params = {"C" : [0.1,0.4,5,10,20,30,40,50]}
svr_cv_model = GridSearchCV(svr_rbf, svr_params, cv=10).fit(X_train, y_train)
print(svr_cv_model.best_params_)


#Fina Model
c_value = pd.Series(svr_cv_model.best_params_)[0]
print(c_value)
svr_tuned_model = SVR("rbf", C = c_value).fit(X_train, y_train)
y_pred_tuned = svr_tuned_model.predict(X_test)
print("Optimum Error:", np.sqrt(mean_squared_error(y_test, y_pred_tuned)))
print("Optimum R2:",r2_score(y_test, y_pred))
