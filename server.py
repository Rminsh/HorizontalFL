from pathlib import Path

import flwr as fl
import numpy as np

# Define the strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,  # Sample 50% of available clients for training
    fraction_evaluate=1,  # Sample 50% of available clients for evaluation
    min_fit_clients=1,  # Minimum number of clients to be sampled for training
    min_evaluate_clients=1,  # Minimum number of clients to be sampled for evaluation
    min_available_clients=1,  # Minimum number of clients that need to be connected
)

# Define the Server Config
config = fl.server.ServerConfig(num_rounds=5)

# Start Flower server
hist = fl.server.start_server(
  server_address="0.0.0.0:8080",
  config=config,
  strategy=strategy
)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
np.save(str(results_dir / "hist.npy"), hist)