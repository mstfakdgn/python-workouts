import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

data = genfromtxt('../DATA/bank_note_data.txt', delimiter=',')

labels = data [:,4]
features = data[:,0:4]

X = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (10,40))
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)


from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import  accuracy_score, confusion_matrix, classification_report

model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(scaled_X_train, y_train, epochs=50, verbose=2)

y_pred = model.predict_classes(scaled_X_test)
print('Accuracy:', accuracy_score(y_pred, y_test))
print('Metrics:', confusion_matrix(y_pred, y_test))
print('Classifaciton Report:', classification_report(y_pred, y_test))

model.save('first_keras_model.h5')

from keras.models import load_model

new_model = load_model('first_keras_model.h5')



