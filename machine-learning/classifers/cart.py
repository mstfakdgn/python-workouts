from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, classification_report, roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier


diabetes = pd.read_csv('../../reading_data/diabetes.csv')
df = diabetes.copy()
df = df.dropna()

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30)

from sklearn.tree import DecisionTreeClassifier

cart = DecisionTreeClassifier()
cart_model = cart.fit(X_train, y_train)


# from skompiler import skompile
# print(skompile(cart_model.predict).to('python/code'))

# x=[500]
# print(((((((0 if x[6] <= 0.6574999988079071 else 1 if x[5] <= 23.550000190734863 else
#     0) if x[5] <= 31.40000057220459 else (1 if x[2] <= 23.0 else ((1 if x[5
#     ] <= 40.60000038146973 else 0) if x[6] <= 0.12849999964237213 else (0 if
#     x[0] <= 2.5 else 1) if x[5] <= 31.800000190734863 else 0 if x[2] <= 
#     83.5 else 1 if x[2] <= 85.5 else 0) if x[6] <= 0.5000000149011612 else 
#     0 if x[0] <= 1.5 else ((1 if x[6] <= 0.5635000169277191 else 1 if x[0] <=
#     2.5 else 0) if x[3] <= 31.0 else 1) if x[5] <= 38.64999961853027 else 0
#     ) if x[6] <= 1.271500051021576 else 1) if x[0] <= 7.0 else 1) if x[5] <=
#     45.39999961853027 else 1 if x[4] <= 180.0 else 0) if x[7] <= 28.5 else 
#     (1 if x[1] <= 22.0 else 0) if x[1] <= 94.5 else (1 if x[5] <= 
#     9.800000190734863 else 0) if x[5] <= 26.949999809265137 else (((((1 if 
#     x[2] <= 65.0 else 0 if x[0] <= 5.5 else 1) if x[7] <= 34.5 else 1) if x
#     [0] <= 7.5 else (1 if x[5] <= 31.15000057220459 else 0 if x[2] <= 70.0 else
#     1) if x[7] <= 41.5 else 0) if x[2] <= 83.0 else 0 if x[3] <= 18.0 else 
#     1) if x[3] <= 27.5 else 0 if x[1] <= 104.0 else (1 if x[7] <= 36.5 else
#     0 if x[3] <= 34.0 else 1 if x[0] <= 10.5 else 0) if x[1] <= 122.5 else 
#     0) if x[6] <= 0.5205000042915344 else (1 if x[5] <= 29.25 else (0 if x[
#     3] <= 9.0 else 1 if x[0] <= 2.5 else 1 if x[3] <= 28.5 else 0) if x[6] <=
#     0.9034999907016754 else 0) if x[0] <= 6.5 else 1) if x[1] <= 127.5 else
#     ((((0 if x[3] <= 28.0 else 1 if x[6] <= 0.3189999982714653 else 0) if x
#     [5] <= 28.199999809265137 else 0 if x[2] <= 76.5 else 1 if x[4] <= 89.5
#      else 0) if x[3] <= 34.5 else 0 if x[7] <= 27.5 else 1) if x[1] <= 
#     151.5 else 0 if x[7] <= 25.5 else (1 if x[5] <= 27.09999942779541 else 
#     1 if x[7] <= 36.5 else 0) if x[7] <= 61.0 else 0) if x[5] <= 
#     29.949999809265137 else 1 if x[2] <= 61.0 else (((((1 if x[2] <= 68.0 else
#     0) if x[6] <= 0.21649999916553497 else 0) if x[7] <= 26.0 else 1 if x[5
#     ] <= 32.75 else 0) if x[5] <= 41.349998474121094 else 1 if x[0] <= 5.0 else
#     0) if x[1] <= 160.0 else 0 if x[2] <= 65.0 else 1 if x[5] <= 
#     46.80000114440918 else 1 if x[4] <= 127.5 else 0) if x[7] <= 28.5 else 
#     (((0 if x[7] <= 36.0 else ((0 if x[6] <= 0.2084999978542328 else 1) if 
#     x[3] <= 16.5 else 0) if x[4] <= 55.0 else 1) if x[1] <= 146.5 else (0 if
#     x[1] <= 157.5 else (1 if x[5] <= 30.699999809265137 else 0) if x[5] <= 
#     33.64999961853027 else 1) if x[0] <= 3.5 else 1 if x[0] <= 9.5 else 1 if
#     x[1] <= 177.0 else 0) if x[6] <= 0.42149999737739563 else ((((0 if x[2] <=
#     69.0 else (0 if x[2] <= 86.0 else 1) if x[1] <= 133.5 else 1) if x[3] <=
#     26.5 else 1) if x[0] <= 12.5 else 0) if x[4] <= 333.5 else 0) if x[5] <=
#     45.95000076293945 else 0) if x[6] <= 1.4070000052452087 else 0))

y_pred = cart_model.predict(X_test)
print('Accuracy:', accuracy_score(y_pred, y_test))


cart_grid = {
    "max_leaf_nodes": range(1, 10),
    "min_samples_split": range(2, 50)
}

# from sklearn.model_selection import GridSearchCV

# cart_cv_model = GridSearchCV(cart_model, cart_grid, cv = 10, verbose=2, n_jobs=-1)
# cart_cv_model.fit(X_train, y_train)

# print(cart_cv_model.best_params_)

cart_tuned = DecisionTreeClassifier(max_leaf_nodes=9, min_samples_split=2).fit(X_train, y_train)
y_tuned_pred = cart_tuned.predict(X_test)
print('Accuracy Tuned :', accuracy_score(y_tuned_pred, y_test))
