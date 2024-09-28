# Binary classification
# DataSet: https://figshare.com/articles/dataset/A_public_data_set_of_overground_and_treadmill_walking_kinematics_and_kinetics_of_healthy_individuals/5722711/4
# Article: Fukuchi, CA., Fukuchi, R.K., Duarte, M., 2018.
# A public dataset of overground and treadmill walking kinematics and kinetics in healthy individuals. PeerJ, 6, e4640
# Features = hip, knee and ankle flexion-extension range of motion
# Label = [0, 1, 2] -> ['comfortable speed', 'fast speed', 'slow speed']

import os
import sys
import glob
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from utilityFunctions import *
import matplotlib.pyplot as plt
import seaborn as sb

# Set the folder path to where the data is located; change accordingly
# Fukuchi et al. (2018) data can be accessed here: https://doi.org/10.6084/m9.figshare.5722711
dataFolder = "C:\\Users\\Nicos\\Documents\\WalkingRunning_Data\\Walking"

# Check if the data folder exists, if not return a useful error message
if not os.path.isdir(dataFolder):
    print(f"The current data folder directory {dataFolder} does not exist.")
    print(f"Set the 'dataFolder' variable to the correct folder.")
    sys.exit()

# Grab the names of all the files
files = os.listdir(dataFolder)

# Separate the files into their categories

# Comfortable walking speed
C_files = glob.glob(os.path.join(dataFolder,"WBDS*walkOCang.txt"))

# Fast walking speed
F_files = glob.glob(os.path.join(dataFolder,"WBDS*walkOFang.txt"))

# Slow walking speed
S_files = glob.glob(os.path.join(dataFolder,"WBDS*walkOSang.txt"))

