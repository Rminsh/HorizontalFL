from pathlib import Path

import flwr as fl
import numpy as np

# Custom aggregation function to average metrics across clients
def weighted_average(metrics):
    num_examples_total = sum([num_examples for num_examples, _ in metrics])
    weighted_metrics = {}
    for num_examples, metric in metrics:
        for k, v in metric.items():
            if k not in weighted_metrics:
                weighted_metrics[k] = 0.0
            weighted_metrics[k] += v * (num_examples / num_examples_total)
    return weighted_metrics

# Define the strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,  # Sample 100% of available clients for training
    fraction_evaluate=1,  # Sample 100% of available clients for evaluation
    min_fit_clients=2,  # Minimum number of clients to be sampled for training
    min_evaluate_clients=2,  # Minimum number of clients to be sampled for evaluation
    min_available_clients=2,  # Minimum number of clients that need to be connected
    evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate evaluation metrics
    fit_metrics_aggregation_fn=weighted_average,  # Aggregate fit metrics
    on_fit_config_fn=lambda r: {"round": r}  # Optionally pass round number
)

# Define the Server Config
config = fl.server.ServerConfig(num_rounds=50)

# Start Flower server
hist = fl.server.start_server(
  server_address="0.0.0.0:8080",
  config=config,
  strategy=strategy
)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
np.save(str(results_dir / "hist.npy"), hist)