from sklearn .ensemble import RandomForestRegressor
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


from sklearn.ensemble import GradientBoostingRegressor

gbm_model = GradientBoostingRegressor()
gbm_model.fit(X_train, y_train)

y_pred = gbm_model.predict(X_test)
print('Error:', np.sqrt(mean_squared_error(y_pred, y_test)))



#Tuning

gbm_params = {
    "learning_rate" : [0.001,0.01,0.1,0.2],
    "max_depth": [3,5,8,50,100],
    "n_estimators":[200,500,1000,2000],
    "subsample" : [1,0.5,0.75]
}

# from sklearn.model_selection import GridSearchCV

# gbm_cv_model = GridSearchCV(gbm_model, gbm_params, cv=10, n_jobs=-1, verbose=2)
# gbm_cv_model.fit(X_train,y_train)
# print(gbm_cv_model.best_params_)


gbm_tuned = GradientBoostingRegressor(learning_rate=0.2, max_depth=3, n_estimators=500, subsample=0.75)
gbm_tuned.fit(X_train, y_train)
y_tuned_pred = gbm_tuned.predict(X_test)
print('Tuned Error:', np.sqrt(mean_squared_error(y_test, y_tuned_pred)))

Importance = pd.DataFrame({"Importance": gbm_tuned.feature_importances_*100},
                          index=X_train.columns)

Importance.sort_values(by = "Importance", axis = 0, ascending= True).plot(kind = "barh", color = "r")
plt.xlabel("Değişken Önem düzeyleri")
plt.show()