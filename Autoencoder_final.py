import time
start = time.time()
import psutil
import os
process = psutil.Process(os.getpid())
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix


top_n_features = 115
# Reading the data from the location and concat them to single dataframe
data =pd.read_csv('./merged_benign.csv')
#importing the fisher score
fisherScore = pd.read_csv('file:///C:/Users/mouni/.spyder-py3/ML/fisher.xlsx')
#importing the top n features using the fisher score
features = fisherScore.iloc[0:int(top_n_features)]['Feature'].values
data = data[list(features)]
#splitting the data into 3 equal parts train,validation and test 
x_train, x_cv, x_test = np.split(data.sample(frac=1, random_state=17), [int(1/3*len(data)), int(2/3*len(data))])
#Using the standar scaler to normalise the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_cv = scaler.fit_transform(x_cv)

#Creating a model
def create_model(input_dim):
    autoencoder = Sequential()
    autoencoder.add(Dense(int(0.75 * input_dim), activation="tanh", input_shape=(input_dim,)))
    autoencoder.add(Dense(int(0.5 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.33 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.25 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.33 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.5 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.75 * input_dim), activation="tanh"))
    autoencoder.add(Dense(input_dim))
    return autoencoder

trained_model = create_model(top_n_features)
trained_model.compile(loss="mean_squared_error",optimizer="sgd")
tensorBoard = TensorBoard(log_dir=f"./logs", histogram_freq=0, write_graph=True, write_images=True)
#Traning the model
trained_model.fit(x_train, x_train, epochs=30, batch_size=64, validation_data=(x_cv, x_cv),
          verbose=1,callbacks=[tensorBoard])
x_cv_predictions = trained_model.predict(x_cv)
mse = np.mean(np.power(x_cv - x_cv_predictions, 2), axis=1)
tr = mse.mean() + mse.std()
print(f"Calculated threshold is:", tr)
end = time.time()
print("time taken for training:", end-start)
start1 = time.time()
#Loading the test data



df_malicious = pd.read_csv('file:///C:/Users/mouni/.spyder-py3/ML/merged_malicious.csv')
#model = AnomalyModel(trained_model, tr, scaler)
df_benign = pd.DataFrame(x_test, columns=data.columns)
df_benign['Anomaly'] = 0
df_malicious = df_malicious.sample(n=df_benign.shape[0], random_state=17)[list(features)]
df_malicious['Anomaly'] = 1
df = df_benign.append(df_malicious)
X_test = df.drop(columns=['Anomaly']).values
X_test_scaled = scaler.transform(X_test)
Y_test = df['Anomaly']
x_pred = trained_model.predict(X_test_scaled)
mse = np.mean(np.power(X_test_scaled - x_pred, 2), axis=1)
Y_pred = (mse > tr)
print(Y_pred)
print('Accuracy')
print(accuracy_score(Y_test, Y_pred))
print('Recall')
print(recall_score(Y_test, Y_pred))
print('Precision')
print(precision_score(Y_test, Y_pred))
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
end1 = time.time()
print("time taken for testing:", end1-start1)
print("memory taken for autoencoder:", process.memory_info().rss)