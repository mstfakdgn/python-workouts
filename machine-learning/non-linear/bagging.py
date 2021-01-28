# creating moren than one decision tree (Combining decision tree models)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler

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

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

scaler.fit(X_test)
X_test = scaler.transform(X_test)

bag_model = BaggingRegressor(bootstrap_features = True)
bag_model.fit(X_train, y_train)
print(bag_model.estimators_)
print(bag_model.estimators_samples_)
print(bag_model.estimators_features_)
print(bag_model.estimators_[1])


#Predict
y_pred = bag_model.predict(X_test)
print('Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Score:', np.sqrt(r2_score(y_pred, y_test)))

# #second tree
# second_tree_predict = bag_model.estimators_[1].fit(X_train, y_train).predict(X_test)
# print('Second Tree Error:', np.sqrt(mean_squared_error(y_test, second_tree_predict)))
# print('Second Tree Score:', np.sqrt(r2_score(y_test, second_tree_predict)))

# #seventh tree
# seventh_tree_predict = bag_model.estimators_[6].fit(X_train, y_train).predict(X_test)
# print('Seventh Tree Error:', np.sqrt(mean_squared_error(y_test, seventh_tree_predict)))
# print('Seventh Tree Score:', np.sqrt(r2_score(y_test, seventh_tree_predict)))




#Tuning
from sklearn.model_selection import GridSearchCV

params = {
    "n_estimators": range(2,20)
}
bag_cv_model = GridSearchCV(bag_model, params, cv=10)
bag_cv_model.fit(X_train, y_train)
print('BEst params:', bag_cv_model.best_params_)


y_tuned_predicted = BaggingRegressor(n_estimators=19).fit(X_train,y_train).predict(X_test)
print('Tuned Error:', np.sqrt(mean_squared_error(y_test, y_tuned_predicted)))
print('Tuned Score:', np.sqrt(r2_score(y_tuned_predicted, y_test)))

