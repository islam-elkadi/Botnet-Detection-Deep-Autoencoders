# General imports
import numpy as np
import pandas as pd
from argparse import ArgumentParser

# Keras imports
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix

class AutoEncoder():

    def __init__(self, input_dim):
        """
            Initializes Deep Autoencoder structure, initialize tensorboard, and initialize Standard Scaler
            Arguments:
                input_dim
        """

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

        # Set EarlyStopping Parameters
        self._early_stopping = EarlyStopping(
        	monitor='loss',
        	min_delta=0.001,
        	patience=4,
        	verbose=0,
        	mode='auto',
        	restore_best_weights=True)
        
        # StandardScaler object
        self._scaler = StandardScaler()

    def train(self, df, batch_size, epochs):
        """
            Trains Deep Autoencoder on training set.
            Arguments:
                df: test dataframe
                learning_rate: 
                batch_size:
                epochs:
        """
        df = self._scaler.fit_transform(df)
        self._autoencoder.compile(loss="mean_squared_error", optimizer="sgd")
        self._autoencoder.fit(df,
                              df,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=1,
                              callbacks=[self._tensorboard, self._early_stopping]
                            )

    def test(self, df):
        """
            Tests performance of Deep Autoencoder on a test set.
            Arguments:
                df: test dataframe
                tr: anomaly threshold
        """
        
        # random shuffle
        df = df.sample(frac=1)

        # Partition test set
        test_input, test_target = df.iloc[:,:-1].values, df.iloc[:,-1].values

        # Scale test input
        test_scaled = self._scaler.fit_transform(test_input)

        # Predict test targets
        test_pred = self._autoencoder.predict(test_scaled)
        mse = np.mean(np.power(test_scaled - test_pred, 2), axis=1)
        predictions = (mse > tr).astype(int)

        print(f"Anomaly threshold: {round(mse, 2)}")
        print(f"Accuracy: {round(accuracy_score(test_target, predictions), 4)*100}%")
        print(f"Recall: {round(recall_score(test_target, predictions), 4)*100}%")
        print(f"Precision: {round(precision_score(test_target, predictions), 4)*100}%")
        print(f"Confusion matrix: {confusion_matrix}")

if __name__=="__main__":

    # CLI arguments
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', help='path to dataset', type=str)
    parser.add_argument('-e', '--epochs', help='No. of epochs', type=int)
    parser.add_argument('-bs', '--batch_size', help='Batch size', type=int)
    # parser.add_argument('-lr', '--learning_rate', help='Learning rate', type=float)
    args = parser.parse_args()

    # Load dataset
    df = pd.concat([x for x in pd.read_csv("dataset.csv", low_memory=False, chunksize=100000)], ignore_index=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')].reset_index(drop=True)

    # Create K-folds 
    kf = KFold(n_splits=10, random_state=42, shuffle=True)

    # Partition begnin dataset
    begnin = df[df["anomaly"]==0]
    begnin_partitions = kf.split(begnin)

    # Partition malicious dataset
    malicious = df[df["anomaly"]==1]
    malicious_partitions = kf.split(malicious)

    # Iterate through begnin & malicious data partitions simultaneously
    for begnin_data, malicious_data in zip(begnin_partitions, malicious_partitions):

        # begnin training, testing set split
        train_idx, test_idx = begnin_data
        begnin_train, begnin_test = begnin.iloc[train_idx,:], begnin.iloc[test_idx,:]

        # malicious testing set split
        train_idx, test_idx = malicious_data
        malicious_test = malicious.iloc[test_idx,:]

        # merge test sets
        merged_test = pd.concat([begnin_test, malicious_test])

        # Initialize auto-encoder
        model = AutoEncoder(115)

        # Train model
        model.train(begnin_train.iloc[:, :-1], args.batch_size, args.epochs)

        # Evaluate model
        model.test(merged_test)