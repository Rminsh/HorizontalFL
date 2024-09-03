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
        
        return serialized_params

    def set_parameters(self, parameters):
        param_keys = self.model.get_params().keys()
        param_dict = {}

        for key, val in zip(param_keys, parameters):
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

            print(f"Set parameter {key} to {param_dict[key]} (type: {type(param_dict[key])})")

        self.model.set_params(**param_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        predicted = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predicted)
        r2 = r2_score(self.y_test, predicted)
        mae = mean_absolute_error(self.y_test, predicted)
        return float(mse), len(self.X_test), {"r2": float(r2), "mae": float(mae)}

# Load data (replace with your actual data loading if different)
data = pd.read_csv("merged_output.csv")
data = data.drop(columns=['timestamp'])
data.fillna(data.mean(), inplace=True)

# Split data into features (X) and target (y)
X  = data.drop(columns=['overall_score','sleep_log_entry_id'])
y = data['overall_score']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using RobustScaler 
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = GradientBoostingRegressor()

# Create a Flower client
client = FlowerClient(model, X_train, X_test, y_train, y_test)

# Start the Flower client
fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())