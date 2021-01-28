import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
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

from sklearn.cross_decomposition import PLSRegression, PLSSVD

pls_model = PLSRegression().fit(X_train, y_train)
# print("all:", pls_model.coef_)
pls_model_six = PLSRegression(n_components=6).fit(X_train, y_train)
# print("6:", pls_model.coef_)
# pls_model = PLSRegression(n_components=2).fit(X_train, y_train)
# print("2:", pls_model.coef_)


# #predict
# print("predictions:", pls_model.predict(X_train))
# print("predictions-six:", pls_model_six.predict(X_train))

y_pred_train = pls_model.predict(X_train)
print("Error Train: ", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("R kare score Train: ", r2_score(y_train, y_pred_train))

y_pred_test = pls_model.predict(X_test)
print("Error Test: ", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("R kare score Test: ", r2_score(y_test, y_pred_test))


#Model tuning
from sklearn import model_selection
import matplotlib.pyplot as plt

cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

RMSE = []
for i in np.arange(1, X_train.shape[1] + 1):
    pls = PLSRegression(n_components=i)
    score = np.sqrt(-1*cross_val_score(pls, X_train, y_train, cv=cv_10, scoring='neg_mean_squared_error').mean())
    RMSE.append(score)

import matplotlib.pyplot as plt

plt.plot(np.arange(1, X_train.shape[1] + 1), np.array(RMSE), '-v', c ="r")
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('PLS Model tuning')
plt.show()

#optimum according to grafic
pls_model_optimum = PLSRegression(n_components=2).fit(X_train, y_train)

y_pred_train = pls_model_optimum.predict(X_train)
print("OPTİMUM Error Train: ", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("OPTİMUM R kare score Train: ", r2_score(y_train, y_pred_train))

y_pred_test = pls_model_optimum.predict(X_test)
print("OPTİMUM Error Test: ", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("OPTİMUM R kare score Test: ", r2_score(y_test, y_pred_test))