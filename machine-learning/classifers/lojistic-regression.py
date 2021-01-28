from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, classification_report, roc_auc_score,roc_curve

diabetes = pd.read_csv('../../reading_data/diabetes.csv')
df = diabetes.copy()
df = df.dropna()

# # Getting to know dataset
# print(df.info())
# print(df.head())
# print(df.describe())

# print(df["Outcome"].value_counts())
# df["Outcome"].value_counts().plot.barh()
# plt.show()

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)
print(X, y)

import statsmodels.api as sm

loj = sm.Logit(y, X)
loj_model = loj.fit()

print(loj_model.summary())


from sklearn.linear_model import LogisticRegression

loj = LogisticRegression(solver="liblinear")
loj_model = loj.fit(X, y)

# print(loj_model.intercept_)
# print(loj_model.coef_)

# y_pred = loj_model.predict(X)
# print("Conclusion Matrix:", confusion_matrix(y, y_pred))
# print("Accuracy Score:", accuracy_score(y, y_pred))
# print('Rapor:', classification_report(y,y_pred))


# print('Probabilities:',loj_model.predict_proba(X)[0:10][:,0:2])
# print('Sonuçlar:', y[0:10])

y_probs = loj_model.predict_proba(X)
y_probs = y_probs[:, 1]

y_pred = [1 if i > 0.5 else 0 for i in y_probs]
print(y_pred[0:10])
print(y[0:10])
print("Conclusion Matrix:", confusion_matrix(y, y_pred))
print("Accuracy Score:", accuracy_score(y, y_pred))
print('Rapor:', classification_report(y,y_pred))

# logit_roc_auc = roc_auc_score(y, loj_model.predict(X))
# fpr, tpr, tresholds = roc_curve(y, loj_model.predict_proba(X)[:,1])
# plt.figure()
# plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
# plt.plot([0,1], [0,1], 'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0,1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)

loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X_train, y_train)
y_pred = loj_model.predict(X_test)
print('Train Test Accracy:', accuracy_score(y_test, y_pred))
print('Cross Val Score:', cross_val_score(loj_model, X_test, y_test, cv=10).mean())