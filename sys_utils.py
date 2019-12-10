#------------------------------------------------------------------------------------------------#
#                                              Imports                                           #
#------------------------------------------------------------------------------------------------#

import json
imporrt argparse
import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from skfeature.function.similarity_based.fisher_score import fisher_score, feature_ranking
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix

#------------------------------------------------------------------------------------------------#
#                                         Global variables                                       #
#------------------------------------------------------------------------------------------------#

scaler = StandardScaler()

tensor_board=TensorBoard(log_dir="./logs/",histogram_freq=0,write_graph=True,write_images=True)