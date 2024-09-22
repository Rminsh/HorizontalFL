# simulation.py
import flwr as fl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from server import weighted_average, CustomFedAvg, TOLERANCE
from client import FlowerClient, set_random_seeds, load_data_for_client, MLP, LinearRegressionModel

# Ensure the 'results' directory exists
os.makedirs('results', exist_ok=True)

# Server Configurations
num_clients_1 = 3
num_clients_2 = 5
num_rounds = 70
details = f"{num_rounds}rounds_"

# Instantiate custom strategies
strategy_1 = CustomFedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
    min_fit_clients=num_clients_1,  # Minimum number of clients to be sampled for training
    min_evaluate_clients=num_clients_1,  # Minimum number of clients to be sampled for evaluation
    min_available_clients=num_clients_1,  # Minimum number of clients that need to be connected
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=weighted_average,
)

strategy_2 = CustomFedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
    min_fit_clients=num_clients_2,  # Minimum number of clients to be sampled for training
    min_evaluate_clients=num_clients_2,  # Minimum number of clients to be sampled for evaluation
    min_available_clients=num_clients_2,  # Minimum number of clients that need to be connected
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=weighted_average,
)

def client_fn_1(cid: str):
    cid_int = int(cid)
    
    set_random_seeds(42)  # Set random seeds for reproducibility

    # Load partitioned data for this client
    train_loader, val_loader, test_loader, input_dim = load_data_for_client(cid_int, num_clients_1)

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Uncomment one of the following lines to choose the model
    # model = LinearRegressionModel(input_dim).to(device)
    model = MLP(input_dim).to(device)

    # Initialize weights (optional for some models)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(init_weights)

    # Create a Flower client
    client = FlowerClient(model, train_loader, val_loader, test_loader, device, cid_int)
    return client

def client_fn_2(cid: str):
    cid_int = int(cid)
    set_random_seeds(42)
    train_loader, val_loader, test_loader, input_dim = load_data_for_client(cid_int, num_clients_2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Uncomment one of the following lines to choose the model
    # model = LinearRegressionModel(input_dim).to(device)
    model = MLP(input_dim).to(device)
    
    # Initialize weights (optional for some models)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(init_weights)
    
    client = FlowerClient(model, train_loader, val_loader, test_loader, device, cid_int)
    return client

# Function to run simulation with a given strategy and number of clients
def run_simulation(client_fn, num_clients, strategy, num_rounds, strategy_name):
    print(f"Starting simulation: {strategy_name} with {num_clients} clients for {num_rounds} rounds.")
    hist = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(str(cid)),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    print(f"Simulation {strategy_name} completed.")
    return hist

# Start Flower server simulations
if __name__ == "__main__":
    # Run simulation for strategy 1
    hist_1 = run_simulation(
        client_fn=client_fn_1,
        num_clients=num_clients_1,
        strategy=strategy_1,
        num_rounds=num_rounds,
        strategy_name="Strategy 1"
    )

    # Run simulation for strategy 2
    hist_2 = run_simulation(
        client_fn=client_fn_2,
        num_clients=num_clients_2,
        strategy=strategy_2,
        num_rounds=num_rounds,
        strategy_name="Strategy 2"
    )

    # Define the number of rounds to exclude
    exclude_rounds = 3  # Number of initial rounds to exclude

    # Ensure there are enough rounds to exclude
    if len(strategy_1.loss_history) <= exclude_rounds or len(strategy_2.loss_history) <= exclude_rounds:
        raise ValueError(f"Not enough rounds to exclude the first {exclude_rounds} rounds.")

    # Define the rounds and corresponding loss/history excluding the first few rounds
    rounds_1 = range(exclude_rounds + 1, len(strategy_1.loss_history) + 1)
    loss_history_1 = strategy_1.loss_history[exclude_rounds:]
    metrics_history_1 = strategy_1.metrics_history[exclude_rounds:]
    fit_metrics_1 = strategy_1.fit_metrics_history[exclude_rounds:]
    train_accuracy_history_1 = strategy_1.train_accuracy_history[exclude_rounds:]
    train_loss_history_1 = strategy_1.train_loss_history[exclude_rounds:]

    rounds_2 = range(exclude_rounds + 1, len(strategy_2.loss_history) + 1)
    loss_history_2 = strategy_2.loss_history[exclude_rounds:]
    metrics_history_2 = strategy_2.metrics_history[exclude_rounds:]
    fit_metrics_2 = strategy_2.fit_metrics_history[exclude_rounds:]
    train_accuracy_history_2 = strategy_2.train_accuracy_history[exclude_rounds:]
    train_loss_history_2 = strategy_2.train_loss_history[exclude_rounds:]

    # Plot Training Accuracy over Rounds
    if train_accuracy_history_1 and train_accuracy_history_2:
        plt.figure(figsize=(10, 5))
        plt.plot(rounds_1, train_accuracy_history_1, marker='o', color='magenta', label=f'Strategy 1: {num_clients_1} Clients')
        plt.plot(rounds_2, train_accuracy_history_2, marker='o', color='pink', label=f'Strategy 2: {num_clients_2} Clients')
        plt.title(f'Global Model Training Accuracy (±{TOLERANCE*100}%) over Rounds')
        plt.xlabel('Round')
        plt.ylabel(f'Training Accuracy (Proportion within ±{TOLERANCE*100}%)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'results/{details}training_accuracy_over_rounds.png')
        plt.show()

    # Plot Training Loss over Rounds
    if train_loss_history_1 and train_loss_history_2:
        plt.figure(figsize=(10, 5))
        plt.plot(rounds_1, train_loss_history_1, marker='o', color='brown', label=f'Strategy 1: {num_clients_1} Clients')
        plt.plot(rounds_2, train_loss_history_2, marker='o', color='darkgoldenrod', label=f'Strategy 2: {num_clients_2} Clients')
        plt.title('Global Model Training Loss over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Training Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'results/{details}training_loss_over_rounds.png')
        plt.show()

    # Plot Test Loss (MSE) over Rounds
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_1, loss_history_1, marker='o', label=f'Strategy 1: {num_clients_1} Clients')
    plt.plot(rounds_2, loss_history_2, marker='o', label=f'Strategy 2: {num_clients_2} Clients')
    plt.title('Global Model Loss (MSE) over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{details}loss_over_rounds.png')
    plt.show()

    # Plot Validation Loss (MSE) from Fit Metrics
    if strategy_1.fit_metrics_history and 'val_mse' in strategy_1.fit_metrics_history[0]:
        val_mse_history_1 = [m['val_mse'] for m in fit_metrics_1]
        val_mse_history_2 = [m['val_mse'] for m in fit_metrics_2]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds_1, val_mse_history_1, marker='o', color='orange', label=f'Strategy 1: {num_clients_1} Clients')
        plt.plot(rounds_2, val_mse_history_2, marker='o', color='darkorange', label=f'Strategy 2: {num_clients_2} Clients')
        plt.title('Global Model Validation Loss (MSE) over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Validation Loss (MSE)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'results/{details}validation_loss_over_rounds.png')
        plt.show()

    # Plot Test MAE over Rounds
    if strategy_1.metrics_history and 'mae' in strategy_1.metrics_history[0]:
        mae_history_1 = [m['mae'] for m in metrics_history_1]
        mae_history_2 = [m['mae'] for m in metrics_history_2]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds_1, mae_history_1, marker='o', color='green', label=f'Strategy 1: {num_clients_1} Clients')
        plt.plot(rounds_2, mae_history_2, marker='o', color='darkgreen', label=f'Strategy 2: {num_clients_2} Clients')
        plt.title('Global Model MAE over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'results/{details}mae_over_rounds.png')
        plt.show()

    # Plot Test MSE over Rounds
    if strategy_1.metrics_history and 'mse' in strategy_1.metrics_history[0]:
        mse_history_1 = [m['mse'] for m in metrics_history_1]
        mse_history_2 = [m['mse'] for m in metrics_history_2]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds_1, mse_history_1, marker='o', color='red', label=f'Strategy 1: {num_clients_1} Clients')
        plt.plot(rounds_2, mse_history_2, marker='o', color='darkred', label=f'Strategy 2: {num_clients_2} Clients')
        plt.title('Global Model MSE over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'results/{details}mse_over_rounds.png')
        plt.show()

    # Plot Test R² Score over Rounds
    if strategy_1.metrics_history and 'r2' in strategy_1.metrics_history[0]:
        r2_history_1 = [m['r2'] for m in metrics_history_1]
        r2_history_2 = [m['r2'] for m in metrics_history_2]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds_1, r2_history_1, marker='o', color='purple', label=f'Strategy 1: {num_clients_1} Clients')
        plt.plot(rounds_2, r2_history_2, marker='o', color='darkviolet', label=f'Strategy 2: {num_clients_2} Clients')
        plt.title('Global Model R² Score over Rounds')
        plt.xlabel('Round')
        plt.ylabel('R² Score')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'results/{details}r2_over_rounds.png')
        plt.show()

    # Plot Validation Accuracy over Rounds
    if strategy_1.fit_metrics_history and 'val_accuracy' in strategy_1.fit_metrics_history[0]:
        val_accuracy_history_1 = [m['val_accuracy'] for m in fit_metrics_1]
        val_accuracy_history_2 = [m['val_accuracy'] for m in fit_metrics_2]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds_1, val_accuracy_history_1, marker='o', color='blue', label=f'Strategy 1: {num_clients_1} Clients')
        plt.plot(rounds_2, val_accuracy_history_2, marker='o', color='cyan', label=f'Strategy 2: {num_clients_2} Clients')
        plt.title(f'Global Model Validation Accuracy (±{TOLERANCE*100}%) over Rounds')
        plt.xlabel('Round')
        plt.ylabel(f'Validation Accuracy (Proportion within ±{TOLERANCE*100}%)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'results/{details}validation_accuracy_over_rounds.png')
        plt.show()

    # Plot Test Accuracy over Rounds
    if strategy_1.metrics_history and 'accuracy' in strategy_1.metrics_history[0]:
        test_accuracy_history_1 = [m['accuracy'] for m in metrics_history_1]
        test_accuracy_history_2 = [m['accuracy'] for m in metrics_history_2]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds_1, test_accuracy_history_1, marker='o', color='teal', label=f'Strategy 1: {num_clients_1} Clients')
        plt.plot(rounds_2, test_accuracy_history_2, marker='o', color='blue', label=f'Strategy 2: {num_clients_2} Clients')
        plt.title(f'Global Model Accuracy over Rounds')
        plt.xlabel('Round')
        plt.ylabel(f'Accuracy')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'results/{details}accuracy_over_rounds.png')
        plt.show()
