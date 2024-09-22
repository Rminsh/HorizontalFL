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
        self.loss_history = []  # Test loss from evaluate
        self.metrics_history = []  # Test metrics from evaluate
        self.fit_metrics_history = []  # Validation metrics from fit
        self.train_loss_history = []  # Training loss from fit
        self.train_accuracy_history = []  # Training accuracy from fit
        self.val_accuracy_history = []  # Validation accuracy from fit

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_fit_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_fit_metrics:
            self.fit_metrics_history.append(aggregated_fit_metrics)
            print(f"Round {rnd} - Aggregated fit metrics: {aggregated_fit_metrics}")
            if 'train_accuracy' in aggregated_fit_metrics:
                self.train_accuracy_history.append(aggregated_fit_metrics['train_accuracy'])
            if 'train_loss' in aggregated_fit_metrics:
                self.train_loss_history.append(aggregated_fit_metrics['train_loss'])
            if 'val_accuracy' in aggregated_fit_metrics:
                self.val_accuracy_history.append(aggregated_fit_metrics['val_accuracy'])
        return aggregated_parameters, aggregated_fit_metrics

    def aggregate_evaluate(self, rnd, results, failures):
        if failures:
            print(f"Round {rnd} had failures: {failures}")
        # Call the original aggregate_evaluate method
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(rnd, results, failures)
        # Store the aggregated loss and metrics
        self.loss_history.append(loss_aggregated)
        self.metrics_history.append(metrics_aggregated)
        print(f"Round {rnd} - Test Loss: {loss_aggregated:.4f}, Test Metrics: {metrics_aggregated}")
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

    # Plot Training Accuracy over Rounds
    if strategy.train_accuracy_history:
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, strategy.train_accuracy_history, marker='o', color='magenta', label='Training Accuracy')
        plt.title(f'Global Model Training Accuracy (±{TOLERANCE*100}%) over Rounds')
        plt.xlabel('Round')
        plt.ylabel(f'Training Accuracy (Proportion within ±{TOLERANCE*100}%)')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/training_accuracy_over_rounds.png')
        plt.show()

    # Plot Training Loss over Rounds
    if strategy.train_loss_history:
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, strategy.train_loss_history, marker='o', color='brown', label='Training Loss')
        plt.title('Global Model Training Loss over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Training Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/training_loss_over_rounds.png')
        plt.show()

    # Plot Test Loss (MSE) over Rounds
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, strategy.loss_history, marker='o', label='Test Loss (MSE)')
    plt.title('Global Model Test Loss (MSE) over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Test Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/test_loss_over_rounds.png')
    plt.show()

    # Plot Validation Loss (MSE) from Fit Metrics
    if strategy.fit_metrics_history and 'val_mse' in strategy.fit_metrics_history[0]:
        val_mse_history = [m['val_mse'] for m in strategy.fit_metrics_history]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, val_mse_history, marker='o', color='orange', label='Validation Loss (MSE)')
        plt.title('Global Model Validation Loss (MSE) over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Validation Loss (MSE)')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/validation_loss_over_rounds.png')
        plt.show()

    # Plot Test MAE over Rounds
    if strategy.metrics_history and 'mae' in strategy.metrics_history[0]:
        mae_history = [m['mae'] for m in strategy.metrics_history]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, mae_history, marker='o', color='green', label='Test MAE')
        plt.title('Global Model Test MAE over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/test_mae_over_rounds.png')
        plt.show()

    # Plot Test MSE over Rounds
    if strategy.metrics_history and 'mse' in strategy.metrics_history[0]:
        mse_history = [m['mse'] for m in strategy.metrics_history]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, mse_history, marker='o', color='red', label='Test MSE')
        plt.title('Global Model Test MSE over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/test_mse_over_rounds.png')
        plt.show()

    # Plot Test R² Score over Rounds
    if strategy.metrics_history and 'r2' in strategy.metrics_history[0]:
        r2_history = [m['r2'] for m in strategy.metrics_history]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, r2_history, marker='o', color='purple', label='Test R² Score')
        plt.title('Global Model Test R² Score over Rounds')
        plt.xlabel('Round')
        plt.ylabel('R² Score')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/test_r2_over_rounds.png')
        plt.show()

    # Plot Validation Accuracy over Rounds
    if strategy.fit_metrics_history and 'val_accuracy' in strategy.fit_metrics_history[0]:
        val_accuracy_history = [m['val_accuracy'] for m in strategy.fit_metrics_history]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, val_accuracy_history, marker='o', color='blue', label=f'Validation Accuracy (±{TOLERANCE*100}%)')
        plt.title(f'Global Model Validation Accuracy (±{TOLERANCE*100}%) over Rounds')
        plt.xlabel('Round')
        plt.ylabel(f'Validation Accuracy (Proportion within ±{TOLERANCE*100}%)')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/validation_accuracy_over_rounds.png')
        plt.show()

    # Plot Test Accuracy over Rounds
    if strategy.metrics_history and 'accuracy' in strategy.metrics_history[0]:
        test_accuracy_history = [m['accuracy'] for m in strategy.metrics_history]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, test_accuracy_history, marker='o', color='cyan', label=f'Test Accuracy (±{TOLERANCE*100}%)')
        plt.title(f'Global Model Test Accuracy (±{TOLERANCE*100}%) over Rounds')
        plt.xlabel('Round')
        plt.ylabel(f'Test Accuracy (Proportion within ±{TOLERANCE*100}%)')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/test_accuracy_over_rounds.png')
        plt.show()
