# General imports
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from timeit import default_timer as timer

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

    def train(self, df, learning_rate, batch_size, epochs):
        """
            Trains Deep Autoencoder on training set.
            Arguments:
                df: test dataframe
                learning_rate: 
                batch_size:
                epochs:
        """
        self._autoencoder.compile(loss="mean_squared_error", optimizer=SGD(lr=learning_rate))
        self._autoencoder.fit(df,
                              df,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=1,
                              callbacks=[self._tensorboard, self._early_stopping]
                            )

    def test(self, df, tr):
        """
            Tests performance of Deep Autoencoder on a test set.
            Arguments:
                df: test dataframe
                tr: anomaly threshold
        """
        
        test_scaled = df.sample(frac=1)
        
        # Partition test set
        test_input, test_target = test_scaled.iloc[:,:-1].values, test_scaled.iloc[:,-1].values

        # Predict test targets
        test_pred = self._autoencoder.predict(test_input)
        mse = np.mean(np.power(test_scaled - test_pred, 2), axis=1)
        predictions = (mse > tr).astype(int)

        print(f"Accuracy: {round(accuracy_score(test_set.iloc[:,-1], predictions), 4)*100}%")
        print(f"Recall: {round(recall_score(test_set.iloc[:,-1], predictions), 4)*100}%")
        print(f"Precision: {round(precision_score(test_set.iloc[:,-1], predictions), 4)*100}%")

if __name__=="__main__":

    # CLI arguments
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', help='No. of epochs', type=int)
    parser.add_argument('-bs', '--batch_size', help='Batch size', type=int)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', type=float)
    parser.add_argument('-tr', '--threshold', help='anomaly threshold', type=float)
    parser.add_argument('-bp', '--begnin_path', help='path to begnin dataset', type=str)
    parser.add_argument('-mp', '--malicious_path', help='path to malicious dataset', type=str)
    args = parser.parse_args()

    # StandardScaler scaler object
    scaler = StandardScaler()

    # Load begnin dataset
    begnin = pd.read_csv(args.begnin_path, low_memory=False)
    begnin = begnin.loc[:, ~begnin.columns.str.contains('^Unnamed')]

    # Load malicious dataset
    malicious = pd.read_csv(args.malicious_path, low_memory=False)
    malicious = malicious.loc[:, ~malicious.columns.str.contains('^Unnamed')]    

    # Create K-folds 
    kf = KFold(n_splits=10, random_state=42, shuffle=True)

    # Partition begnin dataset
    begnin_partitions = kf.split(begnin)

    # Partition malicious dataset
    malicious_partitions = kf.split(malicious)

    # Iterate through begnin & malicious data partitions simultaneously
    for begnin_data, malicious_data in zip(begnin_partitions, malicious_partitions):

        # begnin training, testing set split
        train_idx, test_idx = begnin_data
        begnin_train, begnin_test = begnin.iloc[train_idx,:], begnin.iloc[test_idx,:]

        # malicious training, testing set split
        train_idx, test_idx = malicious_data
        _, malicious_test = malicious.iloc[train_idx,:], malicious.iloc[test_idx,:]

        # scale training_x set
        begnin_train_scaled = scaler.fit_transform(begnin_train)

        # define targets for begning & malicious
        begnin_test["anomaly"] = 0
        malicious_test["anomaly"] = 1

        # merge & scale test sets
        merged_test = pd.concat([begnin_test, malicious_test])
        merged_test.iloc[:, :-1] = scaler.transform(merged_test.iloc[:, :-1])

        # Instantiate & initialize auto-encoder
        model = AutoEncoder(115)

        toc = timer()
        # Train model
        model.train(begnin_train_scaled, args.learning_rate, args.batch_size, args.epochs)
        tic = timer()
        print(tic-toc)

        # Evaluate model
        model.test(merged_test, args.threshold)

        break
