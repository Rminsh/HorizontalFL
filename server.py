# server.py
import flwr as fl
import matplotlib.pyplot as plt
import os

# Define the tolerance used for accuracy in plots
TOLERANCE = 0.10  # 10% tolerance

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
        self.accuracy_history = []  # New list to store accuracy over rounds

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_fit_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_fit_metrics:
            self.fit_metrics_history.append(aggregated_fit_metrics)
            print(f"Round {rnd} - Aggregated fit metrics: {aggregated_fit_metrics}")
            if 'val_accuracy' in aggregated_fit_metrics:
                self.accuracy_history.append(aggregated_fit_metrics['val_accuracy'])
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

# Ensure the 'results' directory exists
os.makedirs('results', exist_ok=True)

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
config = fl.server.ServerConfig(num_rounds=20)  # Adjust the number of rounds as needed

# Start Flower server
if __name__ == "__main__":
    # Start the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy
    )

    # After training, plot the loss and metrics
    rounds = range(1, len(strategy.loss_history) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, strategy.loss_history, marker='o', label='Loss (MSE)')
    plt.title('Global Model Loss over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/loss_over_rounds.png')
    plt.show()

    # Plot Validation MAE
    if strategy.metrics_history and 'mae' in strategy.metrics_history[0]:
        mae_history = [m['mae'] for m in strategy.metrics_history]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, mae_history, marker='o', color='orange', label='Mean Absolute Error (MAE)')
        plt.title('Global Model Validation MAE over Rounds')
        plt.xlabel('Round')
        plt.ylabel('MAE')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/mae_over_rounds.png')
        plt.show()

    # Plot Validation R² Score
    if strategy.metrics_history and 'r2' in strategy.metrics_history[0]:
        r2_history = [m['r2'] for m in strategy.metrics_history]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, r2_history, marker='o', color='green', label='R² Score')
        plt.title('Global Model Validation R² Score over Rounds')
        plt.xlabel('Round')
        plt.ylabel('R² Score')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/r2_over_rounds.png')
        plt.show()

    # Plot Validation Accuracy
    if hasattr(strategy, 'accuracy_history') and strategy.accuracy_history:
        accuracy_history = strategy.accuracy_history
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, accuracy_history, marker='o', color='blue', label=f'Accuracy (±{TOLERANCE*100}%)')
        plt.title(f'Global Model Validation Accuracy (±{TOLERANCE*100}%) over Rounds')
        plt.xlabel('Round')
        plt.ylabel(f'Accuracy (Proportion within ±{TOLERANCE*100}%)')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/accuracy_over_rounds.png')
        plt.show()
