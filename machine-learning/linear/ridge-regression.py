# veri bilimi ödevi için https://www.kaggle.com/floser/hitters

# aşırı öğrenmeye karşıdirençli
# yanlıdır ancak varyansı düşüktür. (Bazen yanlı modelleri daha çok tercih ederiz)
# çok fazla parametre olduğunda EKK ya göre daha iyidir
# çok boyutluluk lanetine çözüm sunar
# çoklu doğrusal bağlantı problemi olduğunda etkilidir
# tüm değişkenler ile model kurar ilgisiz değişkenleri modelden çıkarmaz katsayılarını sıfıra yaklaştırır
# lambda kritik roldedir. iki terimin(formüldeki) göreceli etkilerini kontrol etmeyi sağlar
# lambda için iyi bir değer bulunması önemlidir  Bunun için CV yöntemi kullanılır

import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

hit = pd.read_csv('../../reading_data/Hitters.csv')
df = hit.copy()
df = df.dropna()

## sns.boxplot(x="AtBat", y="Salary", data=df)
## sns.boxplot(x="Hits", y="Salary", data=df)
# sns.boxplot(x="HmRun", y="Salary", data=df)
# sns.boxplot(x="Runs", y="Salary", data=df)
# sns.boxplot(x="Years", y="Salary", data=df)
## sns.boxplot(x="RBI", y="Salary", data=df)
#plt.show()

# print(df.describe().T)
# print(df.corr(method ='pearson'))

df_int = df.select_dtypes(include = ['float64', 'int64'])
Q1 = df_int.quantile(0.25)
Q3 = df_int.quantile(0.75)
IQR = Q3 - Q1
# print(Q1,"\n", Q3,"\n", IQR)

for i in df_int:
    Q1= df[i].quantile(0.25) 
    Q3= df[i].quantile(0.75)
    IQR = Q3 - Q1
    bottom_line = Q1 - 1.5*IQR
    upper_line = Q3 + 1.5*IQR

    # isOutlier =  ((df[i] < bottom_line) | (df[i] > upper_line))
    
    # outliers = df[i][isOutlier]
    # outlierIndexes = df[i][isOutlier].index
    # print(outliers, outlierIndexes)

    df[i][df[i] < bottom_line] = bottom_line
    df[i][df[i] > upper_line] = upper_line


ms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
y = df["Salary"]
# X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis = 1).astype('float64','int64')
X = pd.concat([df_int, ms[["League_N", "Division_W", "NewLeague_N"]]], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.25, random_state=42)
# print(X_train, X_test, y_train, y_test)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
model = reg.fit(X_train, y_train)
predicted_y = model.predict(X_test)
print("Linear Regresyon Error Test: ", np.sqrt(mean_squared_error(y_test, predicted_y)))
print("Linear Regresyon kare score Test: ", r2_score(y_test, predicted_y))

from sklearn.linear_model import Ridge

# ridge_model = Ridge(alpha = 0.1).fit(X_train,y_train)
# # print(ridge_model.coef_)

lambdas = 10**np.linspace(10,-2,100)*0.5

ridge_model = Ridge()
parameters = []

for i in lambdas:
    ridge_model.set_params(alpha = i)
    ridge_model.fit(X_train, y_train)
    parameters.append(ridge_model.coef_)

ax = plt.gca()
ax.plot(lambdas, parameters)
ax.set_xscale('log')

plt.xlabel('Lambda(alpha)')
plt.ylabel('Parameters')
plt.title('Ridge Parameters')
plt.show()





#predict
y_pred_test = ridge_model.predict(X_test)
print("Error Test: ", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("R kare score Test: ", r2_score(y_test, y_pred_test))



#model tuning we need to find optimum lambda value
from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas = lambdas, scoring="neg_mean_squared_error", normalize = True)
ridge_cv.fit(X_train, y_train)
#optimum lambda(alpha) value
print("Optimum Lambda Değeri:", ridge_cv.alpha_)

#tuned model
ridge_tuned = Ridge(alpha = ridge_cv.alpha_, normalize=True).fit(X_train, y_train)

print('tuned Error Test:', np.sqrt(mean_squared_error(y_test, ridge_tuned.predict(X_test))))
print("tuned R kare score Test: ", r2_score(y_test, ridge_tuned.predict(X_test)))