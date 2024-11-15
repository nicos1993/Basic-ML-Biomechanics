# Linear regression
# DataSet: https://zenodo.org/records/13788592
# DataSet authors: Ruth, PS., Uhlrich, S., de Monts, C., Falisse, A., Muccini, J.,
# Covitz, S., Vogt-Domke, S., Karman, L., Ismail., S., Ataide, P., Ong, C.,
# Day, J., Duong, T., Delp, S. 2024
# Kinematics and timed function tests of facioscapulohumeral muscular dystrophy
# and myotonic dystrophy

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utilityFunctions import *

# Set the folder path to where the data is located; change accordingly
dataFolder = "C:\\Users\\Nicos\\Documents\\FHSD_Data"

# Check if the data folder exists, if not return error message
if not os.path.isdir(dataFolder):
    print(f"The current data folder directory {dataFolder} does not exist.")
    print(f"Set the 'dataFolder' variable to the correct folder.")
    sys.exit()

# Data file
dataFile = "video_features.csv"

# Load the data
data = pd.read_csv(os.path.join(dataFolder,dataFile))

# Remove duplicate instances of participants in data based on subject ID, keep first occurence
data.drop_duplicates(subset=['ID'],keep='first',inplace=True)

# Data features can be found from:
# data.columns

# The number of participants (rows) and ~features~ (columns) can be found from:
# data.shape

# Reinitialize a data frame with the necessary features (predictors and outcome)
features = ['10mwrt_speed','10mwrt_com_sway','tug_cone_turn_avel','tug_cone_time']
data = data[features]

# Convert from data frame object to numpy array
data_np = data.to_numpy()

# Downcast the data to be compatible with PyTorch
data_np = torch.from_numpy(data_np).type(torch.float32)

# Separate data into features and outcome
data_X = data_np[:,:-1]
data_y = data_np[:,-1]

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.15, random_state=42)

X_mean = X_train.mean(axis=0)
X_std  = X_train.std(axis=0)
X_train_norm = (X_train - X_mean) / X_std

y_mean = y_train.mean(axis=0)
y_std  = y_train.std(axis=0)
y_train_norm = (y_train - y_mean) / y_std

X_test_norm = (X_test - X_mean) / X_std
y_test_norm = (y_test - y_mean) / y_std

training_data = CustomDataset(labels_file=y_train_norm, features_file=X_train_norm)

training_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)

# layers = []
#
# layers.append(nn.Linear(4,2))
# layers.append(nn.ReLU())
#
# print(layers)
#
# print(*layers)
# model = nn.Sequential()

for inputs, labels in training_dataloader:
    print(labels)


print('break point')

