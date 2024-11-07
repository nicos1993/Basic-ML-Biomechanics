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

# Set the folder path to where the data is located; change accordingly
dataFolder = "C:\\Users\\Nicos\\Documents\\FHSD_Data"

# Data file
dataFile = "video_features.csv"

# Load the data
data = pd.read_csv(os.path.join(dataFolder,dataFile))

# Remove duplicate instances of participants in data based on subject ID, keep first occurence
data.drop_duplicates(subset=['ID'],keep='first',inplace=True)

# Data features can be found from:
# data.columns

# The number of participants (rows) and ~features~ (column) can be found from:
# data.shape

# Check if the data folder exists, if not return error message
if not os.path.isdir(dataFolder):
    print(f"The current data folder directory {dataFolder} does not exist.")
    print(f"Set the 'dataFolder' variable to the correct folder.")
    sys.exit()

