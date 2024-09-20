# client.py
import argparse
import flwr as fl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Function to set random seeds for reproducibility
def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# MLP model
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Linear Regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        out = self.linear(x)
        return out

# Flower client using PyTorch
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader, device, client_id):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.client_id = client_id
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        self.num_epochs = 40

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            avg_epoch_loss = epoch_loss / len(self.train_loader.dataset)
            if (epoch + 1) % 10 == 0:
                print(f"Client {self.client_id} - Epoch {epoch+1}/{self.num_epochs} - Training loss: {avg_epoch_loss:.4f}")
        
        # After training, compute validation metrics
        val_mse, val_mae, val_r2 = self.evaluate_metrics(self.val_loader)
        # Return the updated model parameters
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "val_mse": val_mse,
            "val_mae": val_mae,
            "val_r2": val_r2
        }

    def evaluate_metrics(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                all_outputs.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())
        mse = total_loss / len(data_loader.dataset)
        mae = mean_absolute_error(all_targets, all_outputs)
        r2 = r2_score(all_targets, all_outputs)
        # Convert metrics to standard Python float
        mse = float(mse)
        mae = float(mae)
        r2 = float(r2)
        return mse, mae, r2
        
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                all_outputs.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())
        mse = total_loss / len(self.test_loader.dataset)
        mae = mean_absolute_error(all_targets, all_outputs)
        r2 = r2_score(all_targets, all_outputs)
        # Convert metrics to standard Python float
        mse = float(mse)
        mae = float(mae)
        r2 = float(r2)
        print(f"Client {self.client_id} - Evaluation MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        return mse, len(self.test_loader.dataset), {
            "mse": mse,
            "mae": mae,
            "r2": r2
        }

# Function to load partitioned data for each client
def load_data_for_client(client_id, num_clients):
    # Load the full dataset
    data = pd.read_csv("merged_output.csv")
    data = data.drop(columns=['timestamp', 'sleep_log_entry_id'])

    # Impute missing values using SimpleImputer with 'mean' strategy
    imputer = SimpleImputer(strategy='mean')
    X = data.drop(columns=['overall_score'])
    X = imputer.fit_transform(X)
    y = data['overall_score'].values

    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Combine X and y back into a DataFrame for consistent indexing
    data_processed = pd.DataFrame(X)
    data_processed['overall_score'] = y

    # Shuffle the data
    data_processed = data_processed.sample(frac=1, random_state=42).reset_index(drop=True)

    # Partition the dataset for the client (IID)
    client_indices = list(range(client_id, len(data_processed), num_clients))
    client_data = data_processed.iloc[client_indices]

    X_client = client_data.drop(columns=['overall_score']).values
    y_client = client_data['overall_score'].values

    # Split into training, validation, and testing sets for the client
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_client, y_client, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=42
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    print(f"Client {client_id} - Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, X_train.shape[1]

# Start Flower client with client index and total number of clients
if __name__ == "__main__":
    set_random_seeds(42)  # Set random seeds for reproducibility

    parser = argparse.ArgumentParser()
    parser.add_argument('--client_index', dest='client_index', type=str, help='Client index')
    args = parser.parse_args()

    # Simulate client ID
    client_index = int(args.client_index)  # Assign a unique ID to each client (e.g., 0, 1, 2)
    num_clients = 2  # Define the number of clients

    # Load partitioned data for this client
    train_loader, val_loader, test_loader, input_dim = load_data_for_client(client_index, num_clients)

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model = LinearRegressionModel(input_dim).to(device)
    model = MLP(input_dim).to(device)

    # Create a Flower client
    client = FlowerClient(model, train_loader, val_loader, test_loader, device, client_index)

    # Start the Flower client
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
