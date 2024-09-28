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

# Extract the flexion-extension joint angles, calculate range of motion
C_data = extractData(C_files)
F_data = extractData(F_files)
S_data = extractData(S_files)

# Create classification labels
C_labels = np.ones((len(C_files),1))*0
F_labels = np.ones((len(F_files),1))*1
S_labels = np.ones((len(S_files),1))*2

# Find NANs and remove
C_data, C_labels = checkDataArray(C_data,C_labels)
F_data, F_labels = checkDataArray(F_data,F_labels)
S_data, S_labels = checkDataArray(S_data,S_labels)

# Merge the data and labels, vertically
data   = np.vstack((C_data,F_data,S_data))
labels = np.vstack((C_labels,F_labels,S_labels))

# Normalize each of the input features based on the maximum contained ranges
data_norm = data / np.max(data,axis=0)

# Downcast to be compatible with Pytorch
data_norm = torch.from_numpy(data_norm).type(torch.float32)
labels    = torch.from_numpy(labels).type(torch.long)

# Split data into train and test sets
# Test data 15% of entire data
# Random state set for reproducibility purposes
X_train, X_test, y_train, y_test = train_test_split(data_norm,labels,test_size=0.15,random_state=42)

# Generate a validation testing set from the training set
n_valid_files = np.int32(np.round(X_train.size(dim=0)*0.15)) # 15% of the training set
np.random.seed(1)
valid_file_ints = np.random.randint(0,X_train.size(dim=0),n_valid_files)
X_valid = X_train[valid_file_ints,:]
X_train = np.delete(X_train,valid_file_ints,axis=0)
y_valid = y_train[valid_file_ints,:]
y_train = np.delete(y_train,valid_file_ints,axis=0)

training_data = CustomDataset(labels_file=y_train,features_file=X_train)
testing_data  = CustomDataset(labels_file=y_test,features_file=X_test)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)

# Define hyperparameters
input_size  = 3
output_size = 3
hidden_size = 16

torch.manual_seed(42)