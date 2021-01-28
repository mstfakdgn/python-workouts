# #Simple Linear Regression
# import pandas as pd

# ad = pd.read_csv("../reading_data/Advertising.csv", usecols=[1,2,3,4])
# df = ad.copy()
# # print(df.head())

# # #index came as attribute get rid of that
# # df = df.iloc[:,1:len(df)]
# # print(df.head())

# # print(df.info())
# # print(df.describe().T)
# # print(df.isnull().values.any()) # is there empyty values
# # print(df.corr()) # corrolation


# #ploting
# import seaborn as sns
# import matplotlib.pyplot as plt


# # #Model with taking attributes first
# # import statsmodels.api as sm

# # X = df[["TV"]]
# # print(X)

# # X = sm.add_constant(X)
# # print(X)

# # y = df["sales"]

# # linearModel = sm.OLS(y,X)
# # model = linearModel.fit()
# # print(model.summary())


# # # #Model with picking attributes in later
# # # import statsmodels.formula.api as smf

# # # lm = smf.ols("sales ~ TV", df)
# # # model = lm.fit()
# # # print(model.summary())


# # print(model.params)
# # print(model.summary().tables[1])
# # print(model.conf_int())
# # print(model.f_pvalue, "f_pvalue: ","%.3f" % model.f_pvalue)
# # print("fvalue: ","%.2f" % model.fvalue)
# # print("tvalue: ", "%.2f" % model.tvalues[0:1])
# # print(model.mse_model)
# # print(model.rsquared)
# # print(model.rsquared_adj)

# # print(model.fittedvalues[0:5])
# # print(y[0:5])

# # #Mathemetical equation
# # print("Sales = " + str("%.2f" % model.params[0] + "+ TV" + "*" + str("%.2f" % model.params[1])))

# # #all relations
# # sns.pairplot(df, kind = "reg")
# # plt.show()
# # #couple or pair corrr
# # sns.jointplot(x = "TV", y = "sales", data = df , kind="reg")
# # plt.show()

# # g = sns.regplot(df["TV"], df["sales"], ci=None, scatter_kws={'color':'r', 's':9})
# # g.set_title("Model equation : Sales = 7.03 + TV+0.05")
# # g.set_ylabel("Sales")
# # g.set_xlabel("TV")
# # plt.xlim(-10,310)
# # plt.ylim(bottom=0)

# # plt.show()




# #Lineer Regression with sklearn
# from sklearn.linear_model import LinearRegression

# X = df[["TV"]]
# y= df["sales"]

# reg = LinearRegression()
# model = reg.fit(X, y)
# print(model.intercept_)
# print(model.coef_)
# print(model.score(X,y))
# print(model.predict(X))





# #Gusessing Question=> 30 birim tv harcaması olduğunda satışların tahmini değeri nedir
# #7,03+30*0,04 = 8.23
# print(model.predict([[30]]))
# newData = [[5],[90],[200]]
# print(model.predict(newData))



























# #Importance of leftovers in machine learning
# import pandas as pd
# import statsmodels.api as sm
# import numpy as np
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

# ad = pd.read_csv("../reading_data/Advertising.csv", usecols=[1,2,3,4])
# df = ad.copy()
# print(df.head())

# X = df[["TV"]]
# print(X)

# X = sm.add_constant(X)
# print(X)

# y = df["sales"]

# from sklearn.metrics import mean_squared_error, r2_score
# import statsmodels.formula.api as smf

# lineerModel = smf.ols("sales ~ TV", df)
# model = lineerModel.fit()
# print(model.summary())

# #sum of the squares of differences between predicted values and real values
# mse = mean_squared_error(y, model.fittedvalues)
# print(mse)

# rmse = np.sqrt(mse)
# print(rmse)


# # #with sklearn
# # reg = LinearRegression()
# # model = reg.fit(X,y)

# # print(model.intercept_)
# # print(model.coef_)
# # print(model.score(X,y))
# # print(model.predict(X)[0:10])
# # print(y[0:10])

# compare_table = pd.DataFrame({"real_y": y[0:10], "predicted_y": model.predict(X)[0:10]})
# compare_table["error"] = y[0:10] - model.predict(X)[0:10]
# compare_table["error_square"] = compare_table["error"]**2
# print(compare_table)
# print(np.sum(compare_table["error_square"]), np.sqrt(np.mean(compare_table["error_square"])))
# print(model.resid[0:10])
# plt.plot(model.resid)
# plt.show()























# #Multi Regression Model
# import pandas as pd 
# from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np

# ad = pd.read_csv("../reading_data/Advertising.csv", usecols=[1,2,3,4])
# df = ad.copy()
# print(df)

# X = df.drop("sales", axis =1)
# # print(X)
# y = df["sales"]
# # print(y)

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.20, random_state=42)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# training = df.copy()
# print(training.shape)

# # # with statsmodel
# # import statsmodels.api as smt

# # linearModel = smt.OLS(y_train, X_train)
# # model = linearModel.fit()
# # print(model.summary())


# # with sklearn
# from sklearn.linear_model import LinearRegression

# lm = LinearRegression()
# model = lm.fit(X_train, y_train)
# print(model.intercept_)
# print(model.coef_)
# print(model.score(X,y))




# #Predicting
# # Example: 30 TV, 10Radio, 40Newspaper
# newData = [[30],[10],[40]]
# newDataFrame = pd.DataFrame(newData).T
# # print(newDataFrame)

# print("Predict Value:", model.predict(newDataFrame))

# #predict success score
# rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
# print("rmse with y_train:", rmse)

# rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
# print("rmse with y_test:", rmse)































# #Model Tuning
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np
# from sklearn.linear_model import LinearRegression

# ad = pd.read_csv("../reading_data/Advertising.csv", usecols=[1,2,3,4])
# df = ad.copy()
# # print(df)

# X = df.drop("sales", axis =1)
# # print(X)
# y = df["sales"]
# # print(y)

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.20, random_state=144)
# # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# lm = LinearRegression()
# model = lm.fit(X_train, y_train)

# print("train",np.sqrt(mean_squared_error(y_train, model.predict(X_train))))
# print("test",np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
# print(model.score(X_train, y_train))

# #cross validated score
# print("cross:", cross_val_score(model, X, y, cv= 10, scoring ="r2").mean())
# print("cross train:", np.sqrt(-cross_val_score(model, X_train, y_train, cv= 10, scoring ="neg_mean_squared_error")).mean())
# print("cross test:", np.sqrt(-cross_val_score(model, X_test, y_test, cv= 10, scoring ="neg_mean_squared_error")).mean())


































#PCR Regression
# Değişkenlere boyut indirme uygulandıktan sonra çıkan bileşenlere regresyon modeli kurulması fikrine dayanır
# DATAFRAME -> (multilinear regression) -> y ,    DATAFRAME -> (PCR) -> T -> (multilinear regression) -> y

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression 

hit = pd.read_csv("../reading_data/Hitters.csv")
df = hit.copy()

# #fully there is no Nan
# print(df.isnull().sum().any())

#drop rowns that has NaN values
df = df.dropna()
# print(df.head())
# print(df.info())
# print(df.describe().T)

dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
print(dms.head())

y = df["Salary"]
X_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis = 1).astype("float64", "int64")
print(X_.head())

X = pd.concat([X_, dms[["League_N", "Division_W","NewLeague_N"]]], axis = 1)
print(X.head(), y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.25, random_state=145)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
training = df.copy()
print(training.shape)



#PCE
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

pca = PCA()
X_reduced_train = pca.fit_transform(scale(X_train))
print(X_reduced_train[0:1, :])

#first row all attributes explaining percents ordinary meannig with prev attributes
# ne kadar açıkladığına bakılır
print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4)*100)[0:10])



# #Lr
# lm = LinearRegression()
# pcr_model = lm.fit(X_reduced_train, y_train)
# print("pcr:", pcr_model.intercept_)
# print("pcr:", pcr_model.coef_)



# #Predict
# y_pred = pcr_model.predict(X_reduced_train)
# print(np.sqrt(r2_score(y_train, y_pred)))
# print("train error before tune", np.sqrt(mean_squared_error(y_train, y_pred)))

# print(df["Salary"].mean())


# X_reduced_test = pca.fit_transform(scale(X_test))
# y_pred_test = pcr_model.predict(X_reduced_test)
# print("test error before tune", np.sqrt(mean_squared_error(y_test, y_pred_test)))






#Model Tuning
X_reduced_test = pca.fit_transform(scale(X_test))

lm = LinearRegression()

#in single variable error 
pcr_model = lm.fit(X_reduced_train[:,0:1], y_train)
y_pred = pcr_model.predict(X_reduced_test[:,0:1])
print("single attribute: ",np.sqrt(mean_squared_error(y_test, y_pred)))

#in two dimension error
pcr_model = lm.fit(X_reduced_train[:,0:2], y_train)
y_pred = pcr_model.predict(X_reduced_test[:,0:2])
print("tow attribute: ",np.sqrt(mean_squared_error(y_test, y_pred)))

#all attributes train
pcr_model = lm.fit(X_reduced_train, y_train)
y_pred = pcr_model.predict(X_reduced_train)
print("all attributes train error before tune: ",np.sqrt(mean_squared_error(y_train, y_pred)))

#all attributes test
pcr_model = lm.fit(X_reduced_train, y_train)
y_pred = pcr_model.predict(X_reduced_test)
print("all attributes test error before tune: ",np.sqrt(mean_squared_error(y_test, y_pred)))

#Cross Validation
from sklearn import model_selection

cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

RMSE = []
for i in np.arange(1, X_reduced_train.shape[1] + 1):
    score = np.sqrt(-1*model_selection.cross_val_score(lm, X_reduced_train[:,:i], y_train.ravel(), cv=cv_10, scoring='neg_mean_squared_error').mean())
    RMSE.append(score)

import matplotlib.pyplot as plt

plt.plot(RMSE, '-v')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('PCR Model tuning')

pcr_model = lm.fit(X_reduced_train[:,0:17], y_train)
y_pred = pcr_model.predict(X_reduced_train[:,0:17])
print("train error after tune: ",np.sqrt(mean_squared_error(y_train, y_pred)))

y_pred_test = pcr_model.predict(X_reduced_test[:,0:17])
print("test error after tune: ",np.sqrt(mean_squared_error(y_test, y_pred_test)))


plt.show()