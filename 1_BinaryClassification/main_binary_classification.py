# Binary classification
# DataSet: https://figshare.com/articles/dataset/A_public_data_set_of_overground_and_treadmill_walking_kinematics_and_kinetics_of_healthy_individuals/5722711/4
# Article: Fukuchi, CA., Fukuchi, R.K., Duarte, M., 2018.
# A public dataset of overground and treadmill walking kinematics and kinetics in healthy individuals. PeerJ, 6, e4640
# Features = hip, knee and ankle flexion-extension range of motion
# Label = [0, 1] -> ['comfortable speed', 'fast speed']

import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from utilityFunctions import *
import matplotlib.pyplot as plt
import seaborn as sb

# Set the folder path to where the data is located
# Check if the folder exists, if not return a useful error message
dataFolder = "C:\\Users\\Nicos\\Documents\\WalkingRunning_Data\\Walking"

# Grab the names of all the files
files = os.listdir(dataFolder)

# Separate the files into their categories

# Comfortable walking speed
C_files = glob.glob(os.path.join(dataFolder,"WBDS*walkOCang.txt"))

# Fast walking speed
F_files = glob.glob(os.path.join(dataFolder,"WBDS*walkOFang.txt"))

C_data = extractData(C_files)
F_data = extractData(F_files)

# Create classification labels
C_labels = np.ones((len(C_files),1))*0
F_labels = np.ones((len(F_files),1))*1

C_data, C_labels = checkDataArray(C_data,C_labels)
F_data, F_labels = checkDataArray(F_data,F_labels)

# Merge the data and labels, vertically
data   = np.vstack((C_data,F_data))
labels = np.vstack((C_labels,F_labels))

# Normalize each of the input features based on the maximum contained ranges
data_norm = data / np.max(data,axis=0)

# Downcast to be compatible with Pytorch
data_norm = torch.from_numpy(data_norm).type(torch.float32)
labels    = torch.from_numpy(labels).type(torch.long)

# Split data into train and test sets
# Test data 15% of entire data
# Random state set for reproducibility purposes
X_train, X_test, y_train, y_test = train_test_split(data_norm,labels,test_size=0.15,random_state=42)

training_data = CustomDataset(labels_file=y_train,features_file=X_train)
testing_data  = CustomDataset(labels_file=y_test,features_file=X_test)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)

# Define hyperparameters
input_size  = 3
output_size = 2
hidden_size = 16

torch.manual_seed(42)

# Create model
model = MultiClassModel(input_features=input_size, output_features=output_size, hidden_units=hidden_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the number of training epochs
num_epochs = 200

# Define array to store loss at each epoch
epoch_losses = np.zeros((num_epochs,))

# Training loop
for epoch in range(num_epochs):
    model.train() # Set the model to training mode
    running_loss = 0.0

    for inputs, labels in train_dataloader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        outputs_soft = torch.softmax(outputs, dim=1).argmax(dim=1)

        # Compute loss
        loss = criterion(outputs, labels.squeeze())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()*inputs.size(0)

        # Calculate average training loss for the epoch
        epoch_loss = running_loss/len(train_dataloader.dataset)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss: .4f}")

    # Store losses from each epoch
    epoch_losses[epoch] = epoch_loss

# Testing loop
with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        outputs_soft = torch.softmax(outputs, dim=1).argmax(dim=1)

# Confusion matrix
cm = confusion_matrix(labels.squeeze(),outputs_soft)

# Plot confusion matrix, correctly predicted classes are on the diagonal
plt.figure(figsize=(6,4))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1],yticklabels=[0,1])
plt.xlabel('Predicted Class',fontweight='bold',fontsize=16)
plt.ylabel('Actual Class',fontweight='bold',fontsize=16)
plt.title('Confusion Matrix',fontweight='bold',fontsize=16)
plt.show()

plt.figure()
plt.plot(epoch_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()