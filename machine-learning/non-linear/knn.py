#basamaklar
#k komşu sayısını belirle
#bilinmeyen nokta ile diğer tüm noktalar arasındaki uzaklıkları hesapla
#sınıflandırma ise en sık sınıf regersyon ise ortalama değeri tahmin değeri olarak alınır

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
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
knn_model = KNeighborsRegressor().fit(X_train, y_train)
# print(knn_model.n_neighbors)

#Predict
y_pred = knn_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(r2_score(y_test, y_pred))


#Cross validation
RMSE = []

for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    RMSE.append(rmse)
    print("k = ", k, "için RMSE:", rmse)



print("===========================")

#Model Tuning
from sklearn.model_selection import GridSearchCV

#1 den 30 a kadar kdeğerleri
knn_params = {'n_neighbors': np.arange(1,30,1)}

knn = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn, knn_params, cv=10)
knn_cv_model.fit(X_train, y_train)
#Best K parameter
print("Best k value:", knn_cv_model.best_params_["n_neighbors"])


knn_final_model = KNeighborsRegressor(n_neighbors = knn_cv_model.best_params_["n_neighbors"]).fit(X_train, y_train)
y_final_pred = knn_final_model.predict(X_test)
print("final model error:", np.sqrt(mean_squared_error(y_test, y_final_pred)))