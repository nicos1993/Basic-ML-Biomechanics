import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import pandas as pd

class CustomDataset(Dataset):
   def __init__(self, labels_file, features_file, transform=None,
                target_transform=None):
      self.labels_file = labels_file
      self.features_file = features_file
      self.transform = transform
      self.target_transform = target_transform

   def __len__(self):
      return len(self.labels_file)

   def __getitem__(self, idx):
      feature = self.features_file[idx,:]
      label = self.labels_file[idx]
      if self.transform:
         feature = self.transform(feature)
      if self.target_transform:
         label = self.target_transform(label)
      return feature, label

class MultiClassModel(nn.Module):
   def __init__(self, input_features, output_features, hidden_units=8):
      super().__init__()
      self.linear_layer_stack = nn.Sequential(
         nn.Linear(in_features=input_features, out_features=hidden_units),
         nn.ReLU(),
         nn.Linear(in_features=hidden_units, out_features=hidden_units),
         nn.ReLU(),
         nn.Linear(in_features=hidden_units, out_features=hidden_units),
         nn.ReLU(),
         nn.Linear(in_features=hidden_units, out_features=output_features),
      )

   def forward(self, x):
      return self.linear_layer_stack(x)

class AutoEncoder_wParams(nn.Module):
   def __init__(self, input_size, latent_size):
      super().__init__()

      # Encoder
      self.encoder = nn.Sequential(nn.Linear(input_size, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 16),
                                   nn.ReLU(),
                                   nn.Linear(16, latent_size)
                                   )

      # Decoder
      self.decoder = nn.Sequential(nn.Linear(latent_size, 16),
                                   nn.ReLU(),
                                   nn.Linear(16, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, input_size),
                                   nn.Sigmoid()
                                   )
      def forward(self, x):
         encoded = self.encoder(x)
         decoded = self.decoder(encoded)
         return decoded
class MultiClassModel_wParams(nn.Module):
   def __init__(self, input_features, output_features, nodes_per_layer=8, network_layers=2):
      super().__init__()

      layers = []

      # Define the first layer; input -> first layer
      layers.append(nn.Linear(in_features=input_features, out_features=nodes_per_layer))
      layers.append(nn.ReLU())

      # Include the network layers
      for _ in range(network_layers-1):
         layers.append(nn.Linear(in_features=nodes_per_layer,out_features=nodes_per_layer))
         layers.append(nn.ReLU())

      # Define the output later; last layer -> output
      layers.append(nn.Linear(in_features=nodes_per_layer,out_features=output_features))

      self.linear_layer_stack = nn.Sequential(*layers)

   def forward(self, x):
      return self.linear_layer_stack(x)

def extractData(files):
   # Predefine arrays for the feature variables
   hip_array = np.zeros((101, len(files)))
   knee_array = np.zeros((101, len(files)))
   ankle_array = np.zeros((101, len(files)))

   for i, file in enumerate(files):
      data = pd.read_csv(file, sep="\t", header=0)
      hip_array[:, i] = data['RHipAngleZ']
      knee_array[:, i] = data['RKneeAngleZ']
      ankle_array[:, i] = data['RAnkleAngleZ']

   hip_min = np.min(hip_array, axis=0)
   hip_max = np.max(hip_array, axis=0)
   hip_range = np.abs(hip_max - hip_min)

   knee_min = np.min(knee_array, axis=0)
   knee_max = np.max(knee_array, axis=0)
   knee_range = np.abs(knee_max - knee_min)

   ankle_min = np.min(ankle_array, axis=0)
   ankle_max = np.max(ankle_array, axis=0)
   ankle_range = np.abs(ankle_max - ankle_min)

   data_array = np.array([hip_range, knee_range, ankle_range]).T

   return data_array

# Check data arrays for NANs and remove accordingly
def checkDataArray(DataArray,DataLabels):

    contains_nans = np.isnan(DataArray)

    inds = np.array(np.where(contains_nans[:,:] == True))

    if inds.size > 0:
        DataArray = np.delete(DataArray, inds[0,:].T,axis=0)
        DataLabels = np.delete(DataLabels, inds[0,:].T,axis=0)

    return DataArray, DataLabels

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

# Moving average, smooth curve
def smooth_curve(points, factor=0.9):
   smoothed_points = []
   for point in points:
      if smoothed_points:
         previous = smoothed_points[-1]
         smoothed_points.append(previous * factor + point * (1 - factor))
      else:
         smoothed_points.append(point)
   return smoothed_points