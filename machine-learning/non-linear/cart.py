import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.datasets import load_boston
boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

boston['MEDV'] = boston_dataset.target
names = boston_dataset.feature_names

print(boston.head())
print(names)
print(boston.shape)

from sklearn.tree import DecisionTreeRegressor

array = boston.values

X = array[:,0:13]
Y = array[:,13]

print(X)
print(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)


model = DecisionTreeRegressor(max_leaf_nodes = 20)
rt = model.fit(X_train, Y_train)
print(rt)

Y_pred = rt.predict(X_test)
print('Error:', mean_squared_error(Y_test, Y_pred))
print('Score:', r2_score(Y_test, Y_pred))
