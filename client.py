import argparse
import flwr as fl
import pandas as pd
import numpy as np

from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Explicit mappings for parameters
        self.criterion_map = {0: 'friedman_mse', 1: 'squared_error'}
        self.loss_map = {0: 'squared_error'}  # Add other losses if needed

    def get_parameters(self, config):
        params = self.model.get_params()
        serialized_params = []
        
        for key, val in params.items():
            #print(f"get_params      key:  {key}     val:  {val}")
            if key == 'criterion':
                serialized_params.append(np.array([list(self.criterion_map.values()).index(val)], dtype=np.int32))
            elif key == 'loss':
                serialized_params.append(np.array([list(self.loss_map.values()).index(val)], dtype=np.int32))
            elif isinstance(val, (int, float)):
                serialized_params.append(np.array([val], dtype=np.float32))
            elif isinstance(val, str):
                serialized_params.append(np.array([0], dtype=np.int32))  # Placeholder for unknown string
            else:
                serialized_params.append(np.array([0.0], dtype=np.float32))
        
        #print(serialized_params)
        return serialized_params

    def set_parameters(self, parameters):
        param_keys = self.model.get_params().keys()
        param_dict = {}
        # print("Received Parameters from Server:", parameters)  # Debugging log
        for key, val in zip(param_keys, parameters):
            # print(f"key: {key}   val:  {val}")
            if key == 'criterion':
                param_dict[key] = self.criterion_map.get(int(val[0]), 'unknown')
            elif key == 'loss':
                param_dict[key] = self.loss_map.get(int(val[0]), 'unknown')
            elif key == 'init':
                param_dict[key] = None if val[0] == 0.0 else val[0]
            elif key == 'max_features':
                param_dict[key] = None if val[0] == 0.0 else val[0]
            elif key == 'max_leaf_nodes':
                param_dict[key] = None if val[0] == 0.0 else int(val[0])
            elif key == 'n_iter_no_change':
                param_dict[key] = None if val[0] == 0.0 else int(val[0])
            elif key == 'verbose':
                param_dict[key] = int(val[0]) if isinstance(val[0], (float, np.float32)) else val[0]
            elif key == 'warm_start':
                param_dict[key] = bool(val[0]) if isinstance(val[0], (float, np.float32)) else val[0]
            elif key in ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']:
                param_dict[key] = int(val[0])
            elif isinstance(val[0], (np.float32, float, int)):
                param_dict[key] = float(val[0])
            else:
                param_dict[key] = val[0]

            # print(f"Set parameter {key} to {param_dict[key]} (type: {type(param_dict[key])})")

        # param_dict['warm_start'] = True

        self.model.set_params(**param_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Incrementally increase n_estimators
        additional_estimators = 10  # Add 10 more estimators each round
        self.model.set_params(n_estimators=self.model.get_params()["n_estimators"] + additional_estimators)
        
        print(f"Training with {self.model.get_params()['n_estimators']} estimators")

        self.model.fit(self.X_train, self.y_train)
        updated_params = self.get_parameters(config={})
        return updated_params, len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        predicted = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predicted)
        r2 = r2_score(self.y_test, predicted)
        mae = mean_absolute_error(self.y_test, predicted)
        return float(mse), len(self.X_test), {"r2": float(r2), "mae": float(mae)}

# Function to load partitioned data for each client
def load_data_for_client(client_id, num_clients):
    
    # Load the full dataset
    data = pd.read_csv("merged_output.csv")
    data = data.drop(columns=['timestamp'])
    data.fillna(data.mean(), inplace=True)

    # Split the dataset into features (X) and target (y)
    X = data.drop(columns=['overall_score', 'sleep_log_entry_id'])
    y = data['overall_score']

    # Partition the dataset for the client
    partition_size = len(X) // num_clients
    start = client_id * partition_size
    end = (client_id + 1) * partition_size if client_id < num_clients - 1 else len(X)
    
    X_client = X.iloc[start:end]
    y_client = y.iloc[start:end]

    # Split into training and testing sets for the client
    X_train, X_test, y_train, y_test = train_test_split(X_client, y_client, test_size=0.2, random_state=42)
    
    # Scale the data
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"Client {client_id} - Data shape: {X_client.shape}")
    return X_train, X_test, y_train, y_test

# Initialize the model
model = GradientBoostingRegressor(
    n_estimators=100,   # Starting estimators
    learning_rate=0.01, # Lower learning rate for gradual improvement
    max_depth=5,        # Allow deeper trees
    min_samples_split=10,  # Regularization
    min_samples_leaf=4,    # Larger leaf size for better generalization
    subsample=0.8,      # Subsample for stochastic gradient boosting
    warm_start=True     # Enable warm start to add trees incrementally
    # verbose=1
) # Increasing Model Complexity

# Start Flower client with client index and total number of clients
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--client_index', dest='client_index', type=str, help='Add client_index')
    args = parser.parse_args()
    
    # Simulate client ID (In a real federated setup, each client would get a unique ID)
    client_index = args.client_index  # Assign a unique ID to each client (e.g., 0, 1, 2)
    num_clients = 2  # Define the number of clients

    # Load partitioned data for this client
    X_train, X_test, y_train, y_test = load_data_for_client(int(client_index), num_clients)

    # Create a Flower client
    client = FlowerClient(model, X_train, X_test, y_train, y_test)

    # Start the Flower client
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())