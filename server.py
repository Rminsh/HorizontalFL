# server.py
import flwr as fl
import matplotlib.pyplot as plt

# Custom aggregation function to average metrics across clients
def weighted_average(metrics):
    num_examples_total = sum([num_examples for num_examples, _ in metrics])
    weighted_metrics = {}
    for num_examples, metric in metrics:
        for k, v in metric.items():
            # Ensure v is a standard Python float
            v = float(v)
            if k not in weighted_metrics:
                weighted_metrics[k] = 0.0
            weighted_metrics[k] += v * (num_examples / num_examples_total)
    return weighted_metrics

# Define a custom strategy by subclassing FedAvg
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_history = []
        self.metrics_history = []
        self.fit_metrics_history = []

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_fit_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_fit_metrics:
            self.fit_metrics_history.append(aggregated_fit_metrics)
            print(f"Round {rnd} - Aggregated fit metrics: {aggregated_fit_metrics}")
        return aggregated_parameters, aggregated_fit_metrics

    def aggregate_evaluate(self, rnd, results, failures):
        if failures:
            print(f"Round {rnd} had failures: {failures}")
        # Call the original aggregate_evaluate method
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(rnd, results, failures)
        # Store the aggregated loss and metrics
        self.loss_history.append(loss_aggregated)
        self.metrics_history.append(metrics_aggregated)
        print(f"Round {rnd} - Loss: {loss_aggregated:.4f}, Metrics: {metrics_aggregated}")
        return loss_aggregated, metrics_aggregated

# Instantiate the custom strategy
strategy = CustomFedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
    min_fit_clients=2,  # Minimum number of clients to be sampled for training
    min_evaluate_clients=2,  # Minimum number of clients to be sampled for evaluation
    min_available_clients=2,  # Minimum number of clients that need to be connected
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=weighted_average,
)

# Define the Server Config
config = fl.server.ServerConfig(num_rounds=20)  # Keep number of rounds at 20

# Start Flower server
if __name__ == "__main__":
    # Start the server
    hist = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy
    )

    # After training, plot the loss and metrics
    rounds = range(1, len(strategy.loss_history) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, strategy.loss_history, marker='o')
    plt.title('Global Model Loss over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.savefig('results/loss_over_rounds.png')
    plt.show()

    # If metrics like MAE and R² are available
    if strategy.metrics_history and 'mae' in strategy.metrics_history[0]:
        mae_history = [m['mae'] for m in strategy.metrics_history]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, mae_history, marker='o', color='orange')
        plt.title('Global Model MAE over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Mean Absolute Error')
        plt.grid(True)
        plt.savefig('results/mae_over_rounds.png')
        plt.show()

    if strategy.metrics_history and 'r2' in strategy.metrics_history[0]:
        r2_history = [m['r2'] for m in strategy.metrics_history]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, r2_history, marker='o', color='green')
        plt.title('Global Model R² Score over Rounds')
        plt.xlabel('Round')
        plt.ylabel('R² Score')
        plt.grid(True)
        plt.savefig('results/r2_over_rounds.png')
        plt.show()

    if strategy.fit_metrics_history and 'val_mse' in strategy.fit_metrics_history[0]:
        val_mse_history = [m['val_mse'] for m in strategy.fit_metrics_history]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, val_mse_history, marker='o', color='purple')
        plt.title('Global Model Validation MSE over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Validation MSE')
        plt.grid(True)
        plt.savefig('results/val_mse_over_rounds.png')
        plt.show()
