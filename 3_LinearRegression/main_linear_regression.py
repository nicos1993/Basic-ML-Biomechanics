# Linear regression
# DataSet: https://zenodo.org/records/13788592
# DataSet authors: Ruth, PS., Uhlrich, S., de Monts, C., Falisse, A., Muccini, J.,
# Covitz, S., Vogt-Domke, S., Karman, L., Ismail., S., Ataide, P., Ong, C.,
# Day, J., Duong, T., Delp, S. 2024
# Kinematics and timed function tests of facioscapulohumeral muscular dystrophy
# and myotonic dystrophy

import os
import sys
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utilityFunctions import *
import matplotlib.pyplot as plt


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

# Setup K-fold validation
k = 4
num_val_samples = len(X_train) // k
num_epochs = 1000
all_scores = []

n_inputs = data_X.size(dim=1)
# Define model
def init_nn_model(n_inputs):

    nn_model = nn.Sequential(
        nn.Linear(n_inputs, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    return nn_model

# Loss function: Mean Square Error
loss = nn.MSELoss()

epoch_fold_losses = np.zeros((num_epochs,k))
epoch_fold_losses_val = np.zeros((num_epochs,k))
epoch_fold_losses_val_MAE = np.zeros((num_epochs,k))

for i in range(k):
    print('processing fold #', i)
    val_data = X_train_norm[i * num_val_samples: (i + 1) * num_val_samples]
    val_y    = y_train_norm[i * num_val_samples: (i + 1) * num_val_samples]
    print('indices to: ', i * num_val_samples, ' and from: ', (i + 1) * num_val_samples)

    partial_train_data = np.concatenate(
        [X_train_norm[:i * num_val_samples],
         X_train_norm[(i + 1) * num_val_samples:]],
        axis=0)

    partial_y_data = np.concatenate(
        [y_train_norm[:i * num_val_samples],
         y_train_norm[(i + 1) * num_val_samples:]],
        axis=0)

    training_data = CustomDataset(labels_file=partial_y_data, features_file=partial_train_data)
    training_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

    torch.manual_seed(42)

    # Initialize model
    nn_model = init_nn_model(n_inputs)

    # Define optimizer and learning rate
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        nn_model.train()
        running_loss = 0

        for inputs, output in training_dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            model_output = nn_model(inputs)

            # Compute loss
            loss_ = loss(model_output.squeeze(dim=0),output)

            # Backward pass and optimization
            loss_.backward()
            optimizer.step()

            # Update the running loss
            running_loss += loss_.item()

        print(f"K-fold [{i + 1}, Epoch [{epoch + 1}/{num_epochs}], "
              f"Loss: [{running_loss}]")

        epoch_fold_losses[epoch, i] = running_loss
        epoch_fold_losses_val[epoch, i] = loss(nn_model(val_data).squeeze(dim=1),val_y).detach().numpy()
        epoch_fold_losses_val_MAE[epoch, i] = np.mean(np.abs((
                y_mean.detach().numpy()+y_std.detach().numpy()
                *(nn_model(val_data).squeeze(dim=1)-val_y).detach().numpy())))

    all_scores.append(loss(nn_model(val_data).squeeze(dim=1),val_y).detach().numpy())

all_scores = np.array(all_scores)
print(f"Scores from the K-folds: {all_scores}")
print(f"Mean score from the K-folds: {all_scores.mean()}")

# Calculate the mean per epoch MSE across all folds for validation data
mean_epoch_val_MAE = epoch_fold_losses_val_MAE.mean(axis=1)

plt.figure()
plt.plot(range(1, len(mean_epoch_val_MAE) + 1), mean_epoch_val_MAE)
plt.xlabel("Epochs",fontweight='bold',fontsize=16)
plt.ylabel("Validation MSE",fontweight='bold',fontsize=16)
plt.show()

print('break point')

