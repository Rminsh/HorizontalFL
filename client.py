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

    def get_parameters(self, config):
        return self.model.get_params()

    def set_parameters(self, parameters):
        param_dict = dict(zip(self.model.get_params().keys(), parameters))
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
fl.client.start_client(server_address="0.0.0.0:8080", client=client)