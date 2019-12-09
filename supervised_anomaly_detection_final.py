import time
start = time.time()
import psutil
import os
process = psutil.Process(os.getpid())
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix, f1_score

# Reading the data from the location and concat them to single dataframe
df_benign =pd.read_csv('file:///C:/Users/mouni/.spyder-py3/ML/merged_benign.csv')
df_malicious = pd.read_csv('file:///C:/Users/mouni/.spyder-py3/ML/merged_malicious.csv')
df_malicious = df_malicious.sample(n=df_benign.shape[0], random_state=17)
df_benign['Anomaly'] = 0
df_malicious['Anomaly'] = 1
df = df_benign.append(df_malicious)
Y = df['Anomaly']
X = df.drop(columns=['Anomaly']).values
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2)
#scale data for efficiency
scaler =  StandardScaler()
x_train = scaler.fit_transform(x_train)
clf = RFC()
clf.fit(x_train, y_train)
end = time.time()
print("time consumed for training:", end-start)

#test the model on data
start1 = time.time()
x_test = scaler.transform(x_test)
Y_pred = clf.predict(x_test)
print('Accuracy')
print(accuracy_score(y_test, Y_pred))
print('Recall')
print(recall_score(y_test, Y_pred))
print('Precision')
print(precision_score(y_test, Y_pred))
print('f1_score')
print(f1_score(y_test, Y_pred))
cm = confusion_matrix(y_test, Y_pred)
print(cm)
end1 = time.time()
print("time taken for testing:", end1-start1)
print("memory taken for supervised anomaly detection:", process.memory_info().rss)