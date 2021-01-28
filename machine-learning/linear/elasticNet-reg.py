#amaç hata kareler toplamını minimize eden katsayıları bu katsayılara bir ceza uygulayarak bulmaktır l1 ve l2 yaklaşımlarını birleştirir

import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

hit = pd.read_csv('../reading_data/Hitters.csv')
df = hit.copy()
df = df.dropna()
ms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis = 1).astype('float64','int64')
X = pd.concat([X_, ms[["League_N", "Division_W", "NewLeague_N"]]], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.25, random_state=42)
# print(X_train, X_test, y_train, y_test)

#Model
from sklearn.linear_model import ElasticNet

enet_model = ElasticNet().fit(X_train, y_train)
# print(enet_model.coef_)
# print(enet_model.intercept_)

#Predict
y_pred = enet_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(r2_score(y_test, y_pred))


#Model Tuning
from sklearn.linear_model import ElasticNetCV

enet_cv_model = ElasticNetCV(cv = 10, random_state=0).fit(X_train,y_train)
print("optimum alpha:" , enet_cv_model.alpha_)

enet_tuned = ElasticNet(alpha=enet_cv_model.alpha_).fit(X_train, y_train)

y_pred_tuned = enet_tuned.predict(X_test)
print("tuned error:",np.sqrt(mean_squared_error(y_test, y_pred_tuned)))
print("tuned r:",r2_score(y_test, y_pred_tuned))