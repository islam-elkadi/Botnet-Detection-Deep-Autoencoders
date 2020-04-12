# General imports
import numpy as np
import pandas as pd

# Keras imports
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix

class AutoEncoder():

    def __init__(self, input_dim):

        # Initialize self._autoencoder
        self._autoencoder = Sequential()
        self._autoencoder.add(Dense(int(0.75 * input_dim), activation="relu", input_shape=(input_dim,)))
        self._autoencoder.add(Dense(int(0.5 * input_dim), activation="relu"))
        self._autoencoder.add(Dense(int(0.33 * input_dim), activation="relu"))
        self._autoencoder.add(Dense(int(0.25 * input_dim), activation="relu"))
        self._autoencoder.add(Dense(int(0.33 * input_dim), activation="relu"))
        self._autoencoder.add(Dense(int(0.5 * input_dim), activation="relu"))
        self._autoencoder.add(Dense(int(0.75 * input_dim), activation="relu"))
        self._autoencoder.add(Dense(input_dim))

        # Initialize tensorboard
        self._tensorboard = TensorBoard(
            log_dir="logs",
            histogram_freq=0,
            write_graph=True,
            write_images=True)
        
        # StandardScaler object
        self._scaler = StandardScaler()

    def preprocess(self, df):
        
        # Create malicious set
        malicious = df[df["anomaly"]==1]

        # Create & segment begnin set
        benign = df[df["anomaly"]==0]
        benign_train, benign_optimization, benign_test_unscaled = np.split(benign.sample(frac=1, random_state=42), [int(1/3 * len(benign)), int(2/3 * len(benign))])
        benign_train_scaled = self._scaler.fit_transform(benign_train.iloc[:, :-1].values)
        benign_optimization_scaled = self._scaler.fit_transform(benign_optimization.iloc[:, :-1].values)

        return benign_train_scaled, benign_optimization_scaled, benign_test_unscaled, malicious

    def train(self, train_scaled):
        self._autoencoder.compile(loss="mean_squared_error", optimizer="sgd")
        self._autoencoder.fit(train_scaled,
                              train_scaled,
                              epochs=60,
                              batch_size=100,
                              verbose=1,
                              callbacks=[self._tensorboard]
                              )

    def test(self, begnin_validation, begnin_test, malicious_test):

        # Create MSE
        valid_pred = self._autoencoder.predict(begnin_validation)
        mse = np.mean(np.power(begnin_validation - valid_pred, 2), axis =1)

        # Define begnin threshold
        tr = mse.mean() + mse.std()

        # Test model
        test_set = pd.concat([begnin_test, malicious_test], sort=True, ignore_index=True)
        test_scaled = self._scaler.transform(test_set.iloc[:,:-1].values)
        test_pred = self._autoencoder.predict(test_scaled)

        # Predict test set
        mse = np.mean(np.power(test_scaled - test_pred, 2), axis=1)
        predictions = (mse > tr).astype(int)

        print(f"Accuracy: {round(accuracy_score(test_set.iloc[:,-1], predictions), 4)*100}%")
        print(f"Recall: {round(recall_score(test_set.iloc[:,-1], predictions), 4)*100}%")
        print(f"Precision: {round(precision_score(test_set.iloc[:,-1], predictions), 4)*100}%")

if __name__=="__main__":

    # StandardScaler scaler object
    scaler = StandardScaler()

    # Load dataset
    df = pd.concat([x for x in pd.read_csv("dataset.csv", low_memory=False, chunksize=100000)], ignore_index=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')].reset_index(drop=True)

    # Partition dataset
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:,-1], test_size=0.20, random_state=42)

    # Create K-folds 
    kf = KFold(n_splits=10, random_state=42, shuffle=True)

    for train_index, valid_index in kf.split(x_train):

        # create training set
        training_x, _ = x_train[train_index], y_train[train_index]

        # create validation set
        testing_x, testing_y = x_test[valid_index], y_test[valid_index]

        # scale training_x set
        training_x = scaler.fit_transform(training_x)

        # Instantiate & initialize auto-encoder
        model = AutoEncoder(115)

        # Train model
        model.train(training_x)

        # Evaluate model
        model.test(benign_optimization_scaled, begnin_test_unscaled, malicious)
