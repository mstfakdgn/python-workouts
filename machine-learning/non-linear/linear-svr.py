#Support Vektor Regression
#amaç bir marjin aralığına maksimum noktayı en küçük hata ile alabilecek şekilde doğru yada eğriyi belirlemektir (Destek vektor regresyonu)
#Robast algoritmalar aykırı gözlemlere çok değişken problemine daha dayanıklıdır svr aykırı gözlemlere daha dayanıklıdır 

#There is 2 kind of support vector reg linear and non-linear

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

X_train = pd.DataFrame(X_train["Hits"])
X_test = pd.DataFrame(X_test["Hits"])

from sklearn.svm import SVR

svr_model = SVR("linear").fit(X_train, y_train)
svr_pred = svr_model.predict(X_train)
print(svr_pred[0:5])
print("y = {0} + {1} x".format(svr_model.intercept_[0], svr_model.coef_[0][0]))

# plt.scatter(X_train, y_train)
# plt.plot(X_train, svr_pred, color="red")
# plt.show()


from sklearn.linear_model import LinearRegression

lm_model = LinearRegression().fit(X_train, y_train)
lm_pred = lm_model.predict(X_train)
print("y = {0} + {1} x".format(lm_model.intercept_, lm_model.coef_[0]))

print("svr tahmini:", -48.69756097561513 + 4.969512195122093*91)
print("linear model  tahmini:", -8.814095480334572 + 5.1724561354706875*91)

#Fark nereden geldi ?
plt.scatter(X_train, y_train)
plt.plot(X_train, lm_pred, 'g')
plt.plot(X_train, svr_pred, 'r')
plt.xlabel("Hits")
plt.ylabel("Salary")
plt.show()




#Predict
y_pred = svr_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))



#Model Tuning
from sklearn.model_selection import GridSearchCV

svr_params = {"C": np.arange(0.1,2.0,0.1)}
svr_cv_model = GridSearchCV(svr_model, svr_params, cv=10).fit(X_train, y_train)

#Optimum C
print("OPTIMUM: ",svr_cv_model.best_params_)





#tuned
c_value = pd.Series(svr_cv_model.best_params_)
svr_tuned = SVR("linear", C = c_value).fit(X_train, y_train)
y_pred_tuned = svr_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred_tuned)))