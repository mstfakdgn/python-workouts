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


rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print('Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Score:', np.sqrt(r2_score(y_test, y_pred)))


# Tuning
# there is two important variable Decision tree number, dğülerde göz önünde bulunduralacak değişken sayısı

rf_params = {
    'max_depth': list(range(1, 10)),
    'max_features': [3, 5, 10, 15],
    'n_estimators': [100, 200, 500, 1000, 2000]
}

# from sklearn.model_selection import GridSearchCV

# # n_jobs = -1 parameter is to much full performance
# rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1)
# rf_cv_model.fit(X_train, y_train)
# print(rf_cv_model.best_params_)

rf_tuned = RandomForestRegressor(max_depth=8, max_features=3, n_estimators=100)
rf_tuned.fit(X_train, y_train)
y_tuned_pred = rf_tuned.predict(X_test)
print('Tuned Error:', np.sqrt(mean_squared_error(y_test, y_tuned_pred)))
print('Tuned Score:', np.sqrt(r2_score(y_test, y_tuned_pred)))


Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                          index=X_train.columns)

Importance.sort_values(by = "Importance", axis = 0, ascending= True).plot(kind = "barh", color = "r")
plt.xlabel("Değişken Önem düzeyleri")
plt.show()