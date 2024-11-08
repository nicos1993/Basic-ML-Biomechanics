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

# Set the folder path to where the data is located; change accordingly
dataFolder = "C:\\Users\\Nicos\\Documents\\FHSD_Data"

# Data file
dataFile = "video_features.csv"

# Load the data
data = pd.read_csv(os.path.join(dataFolder,dataFile))

# Remove duplicate instances of participants in data based on subject ID, keep first occurence
data.drop_duplicates(subset=['ID'],keep='first',inplace=True)

# Check if the data folder exists, if not return error message
if not os.path.isdir(dataFolder):
    print(f"The current data folder directory {dataFolder} does not exist.")
    print(f"Set the 'dataFolder' variable to the correct folder.")
    sys.exit()

# Data features can be found from:
# data.columns

# The number of participants (rows) and ~features~ (columns) can be found from:
# data.shape

# Reinitialize a data frame with the necessary features (predictors and outcome)
features = ['10mwrt_speed','10mwrt_com_sway','tug_cone_turn_avel','tug_cone_time']
data = data[features]

# Convert from data frame object to numpy array
data_np = data.to_numpy()

print('break point')

