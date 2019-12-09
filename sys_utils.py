#------------------------------------------------------------------------------------------------#
#                                              Imports                                           #
#------------------------------------------------------------------------------------------------#

import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request, make_response
from skfeature.function.similarity_based.fisher_score import fisher_score, feature_ranking
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix

#------------------------------------------------------------------------------------------------#
#                                         Global variables                                       #
#------------------------------------------------------------------------------------------------#

scaler = StandardScaler()