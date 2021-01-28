from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
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

# X_train = pd.DataFrame(X_train["Hits"])
# X_test = pd.DataFrame(X_test["Hits"])


cart_model = DecisionTreeRegressor()
cart_model.fit(X_train, y_train)

# #Visualizing
# X_grid = np.arange(min(np.array(X_train)), max(np.array(X_train)), 0.01)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X_train, y_train, color = 'red')
# plt.plot(X_grid, cart_model.predict(X_grid), color = 'blue')
# plt.title('CART REGRESYON AĞACI')
# plt.xlabel('Hits')
# plt.ylabel('Salary')
# # plt.show()

# # pip install skompiler && pip install astor
# from skompiler import skompile
# print(skompile(cart_model.predict).to('python/code'))


# #Predict
# x= [91]
# print(((920.0 if x[0] <= 18.0 else ((500.0 if x[0] <= 34.5 else (70.0 if x[0] <=
#     38.0 else 175.0) if x[0] <= 39.5 else 90.0 if x[0] <= 40.5 else 67.5) if
#     x[0] <= 41.5 else 900.0 if x[0] <= 42.5 else (((((215.55566666666667 if
#     x[0] <= 44.5 else 180.0) if x[0] <= 46.5 else 347.5 if x[0] <= 48.0 else
#     337.5) if x[0] <= 50.0 else 70.0) if x[0] <= 51.5 else 507.5) if x[0] <=
#     52.5 else 150.0 if x[0] <= 53.5 else 146.83333333333334) if x[0] <=
#     54.5 else 1300.0 if x[0] <= 55.5 else (170.0 if x[0] <= 56.5 else
#     193.75) if x[0] <= 57.5 else ((((((340.0 if x[0] <= 59.0 else 418.5) if
#     x[0] <= 60.5 else 235.0 if x[0] <= 62.0 else 341.667) if x[0] <= 63.5 else
#     75.0) if x[0] <= 64.5 else 650.0) if x[0] <= 65.5 else 170.0 if x[0] <=
#     67.0 else 228.66666666666666) if x[0] <= 69.0 else 472.5) if x[0] <=
#     70.5 else (100.0 if x[0] <= 71.5 else 130.0) if x[0] <= 72.5 else
#     409.16650000000004 if x[0] <= 74.5 else 215.0) if x[0] <= 76.5 else (((
#     505.0 if x[0] <= 77.5 else ((((328.88899999999995 if x[0] <= 79.0 else
#     700.0) if x[0] <= 80.5 else 348.75 if x[0] <= 81.5 else 267.5) if x[0] <=
#     82.5 else 600.0 if x[0] <= 83.5 else 600.0) if x[0] <= 84.5 else (
#     331.25 if x[0] <= 85.5 else (180.0 if x[0] <= 86.5 else 91.5) if x[0] <=
#     88.5 else 450.0 if x[0] <= 90.5 else 125.0) if x[0] <= 91.5 else 411.25 if
#     x[0] <= 92.5 else 250.0) if x[0] <= 93.5 else 670.0 if x[0] <= 94.5 else
#     ((100.0 if x[0] <= 95.5 else 504.16650000000004) if x[0] <= 96.5 else
#     210.0 if x[0] <= 98.0 else 87.5) if x[0] <= 100.0 else 466.0) if x[0] <=
#     101.5 else 247.5 if x[0] <= 102.5 else 257.3334) if x[0] <= 103.5 else
#     (750.0 if x[0] <= 105.0 else 850.0) if x[0] <= 107.0 else 162.5 if x[0] <=
#     109.0 else (560.0 if x[0] <= 111.0 else 442.5 if x[0] <= 112.5 else
#     487.5) if x[0] <= 114.0 else 300.0) if x[0] <= 116.0 else 110.0) if x[0
#     ] <= 117.5 else ((((1300.0 if x[0] <= 118.5 else 773.3333333333334 if x
#     [0] <= 120.5 else 442.5) if x[0] <= 122.5 else 1240.0 if x[0] <= 124.0 else
#     1925.5710000000001) if x[0] <= 125.5 else (561.25 if x[0] <= 126.5 else
#     (695.2776666666667 if x[0] <= 127.5 else 1043.75) if x[0] <= 128.5 else
#     (750.0 if x[0] <= 129.5 else 480.0) if x[0] <= 130.5 else
#     726.6666666666666) if x[0] <= 131.5 else (((611.6665 if x[0] <= 133.5 else
#     461.0) if x[0] <= 135.5 else 725.0) if x[0] <= 137.0 else 152.5 if x[0] <=
#     138.5 else 555.0 if x[0] <= 139.5 else 200.0) if x[0] <= 140.5 else
#     712.5 if x[0] <= 141.5 else 777.5) if x[0] <= 143.0 else (((
#     1021.6666666666666 if x[0] <= 144.5 else 500.0 if x[0] <= 145.5 else
#     815.0) if x[0] <= 146.5 else 1230.0 if x[0] <= 148.0 else 787.5 if x[0] <=
#     149.5 else 1000.0) if x[0] <= 150.5 else 2460.0) if x[0] <= 151.5 else
#     (451.6666666666667 if x[0] <= 153.0 else 580.0 if x[0] <= 155.5 else
#     530.0) if x[0] <= 157.5 else (((((775.0 if x[0] <= 158.5 else 759.1665) if
#     x[0] <= 159.5 else 1670.0 if x[0] <= 161.5 else 923.0 if x[0] <= 165.5 else
#     863.0556666666666) if x[0] <= 168.5 else 743.3333333333334) if x[0] <=
#     169.5 else 1118.75 if x[0] <= 170.5 else 1350.0) if x[0] <= 171.5 else
#     165.0 if x[0] <= 173.0 else (849.3335 if x[0] <= 175.5 else 1350.0) if
#     x[0] <= 177.5 else (740.0 if x[0] <= 178.5 else 575.0 if x[0] <= 181.5 else
#     630.0) if x[0] <= 185.0 else 1300.0 if x[0] <= 198.5 else 740.0) if x[0
#     ] <= 212.0 else 350.0) if x[0] <= 225.5 else 1975.0))


# print(cart_model.predict(X_test)[0:5])
# print(cart_model.predict([[91]]))

y_pred = cart_model.predict(X_test)

print('Error:', mean_squared_error(y_test, y_pred))
print('Score:', r2_score(y_test, y_pred))


# Tuning

params = {
    "min_samples_split": range(2, 100),
    "max_leaf_nodes": range(2, 10)
}

cart_cv_model = GridSearchCV(cart_model, params, cv=10)
cart_cv_model.fit(X_train, y_train)
print(cart_cv_model.best_params_)


cart_tuned_model = DecisionTreeRegressor(
    max_leaf_nodes=9, min_samples_split=37)
cart_tuned_model.fit(X_train, y_train)
y_tuned_pred = cart_tuned_model.predict(X_test)

print('Error:', mean_squared_error(y_test, y_tuned_pred))
print('Score:', r2_score(y_test, y_tuned_pred))
