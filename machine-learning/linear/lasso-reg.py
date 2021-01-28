# ridge regresyonun ilgili-ilgisiz tüm değişkenleri modelde bırakma dezavantajını gidermek için önerilmiştir
# lassoda katsayıları sıfıra yaklaştırır
# fakat L1 formu lambda yeteri kadar büyük olduğunda bazı katsayıları sıfır yapar Böylece değişken seçimi yapmış olur
# lambda nın doğru seçilmesi çok önemlidir. burada da CV kullanılır
# Ridge ve Lasso yöntemleri birbirinden üstün değildir

# Lambda ayar parametresinin seçilmesi
# lambda nın sıfır olduğu yer EKKdır HKT yi minimum yapan lambdayı arıyoruz
# lambda için belirli değerleri içeren bir küme seçilir ve her birisi için cross validation test hatası hesaplanır
# en küçük cross validation ı veren lambda ayar parametresi olarak seçilir
# son olarak seçilen bu lambda ile model yeniden tüm gözlemlere fit edilir


# Model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
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
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'],
             axis=1).astype('float64', 'int64')
X = pd.concat([X_, ms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
# print(X_train, X_test, y_train, y_test)


lasso_model = Lasso(alpha=0.1).fit(X_train, y_train)
# print(lasso_model.coef_)

lasso = Lasso()
lambdas = 10**np.linspace(10, -2, 100)*0.5
parameters = []

for i in lambdas:
    lasso.set_params(alpha=i)
    lasso.fit(X_train, y_train)
    parameters.append(lasso.coef_)

ax = plt.gca()
ax.plot(lambdas*2, parameters)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.show()


# predict
y_pred_test = lasso.predict(X_test)
print("Error Test: ", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("R kare score Test: ", r2_score(y_test, y_pred_test))


# Model Tuning

lasso_cv_model = LassoCV(alphas=None, cv=10, max_iter=10000, normalize=True)
lasso_cv_model.fit(X_train, y_train)
# print(lasso_cv_model.alpha_)

lasso_tuned_model = Lasso(alpha=lasso_cv_model.alpha_)
lasso_tuned_model.fit(X_train, y_train)
pred_tunned_y_test = lasso_tuned_model.predict(X_test)
print("Tuned error:",np.sqrt(mean_squared_error(pred_tunned_y_test, y_test)))
