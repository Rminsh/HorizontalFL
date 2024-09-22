# simulation.py
import flwr as fl
import torch
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
    # model = LinearRegressionModel(input_dim).to(device)
    model = MLP(input_dim).to(device)

    # Create a Flower client
    return FlowerClient(model, train_loader, val_loader, test_loader, device, cid_int).to_client()

def client_fn_2(cid: str):
    cid_int = int(cid)
    set_random_seeds(42)
    train_loader, val_loader, test_loader, input_dim = load_data_for_client(cid_int, num_clients_2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim).to(device)
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
    exclude_rounds = 0  # Number of initial rounds to exclude

    # Ensure there are enough rounds to exclude
    if len(strategy_1.loss_history) <= exclude_rounds or len(strategy_2.loss_history) <= exclude_rounds:
        raise ValueError(f"Not enough rounds to exclude the first {exclude_rounds} rounds.")

    # Define the rounds and corresponding loss/history excluding the first few rounds
    rounds_1 = range(exclude_rounds + 1, len(strategy_1.loss_history) + 1)
    loss_history_1 = strategy_1.loss_history[exclude_rounds:]
    accuracy_history_1 = strategy_1.accuracy_history[exclude_rounds:]

    rounds_2 = range(exclude_rounds + 1, len(strategy_2.loss_history) + 1)
    loss_history_2 = strategy_2.loss_history[exclude_rounds:]
    accuracy_history_2 = strategy_2.accuracy_history[exclude_rounds:]

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_1, loss_history_1, marker='o', label=f'{num_clients_1} Clients')
    plt.plot(rounds_2, loss_history_2, marker='o', label=f'{num_clients_2} Clients')
    plt.title('Global Model Loss over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{details}loss_over_rounds.png')
    plt.show()

    # Plot Validation MAE
    if strategy_1.metrics_history and 'mae' in strategy_1.metrics_history[0]:
        mae_history_1 = [m['mae'] for m in strategy_1.metrics_history]
        mae_history_1 = mae_history_1[exclude_rounds:]
        mae_history_2 = [m['mae'] for m in strategy_2.metrics_history]
        mae_history_2 = mae_history_2[exclude_rounds:]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds_1, mae_history_1, marker='o', color='orange', label=f'{num_clients_1} Clients')
        plt.plot(rounds_2, mae_history_2, marker='o', color='indigo', label=f'{num_clients_2} Clients')
        plt.title('Global Model Validation MAE over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Mean Absolute Error')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'results/{details}mae_over_rounds.png')
        plt.show()

    # Plot Validation R² Score
    if strategy_1.metrics_history and 'r2' in strategy_1.metrics_history[0]:
        r2_history_1 = [m['r2'] for m in strategy_1.metrics_history]
        r2_history_1 = r2_history_1[exclude_rounds:]
        r2_history_2 = [m['r2'] for m in strategy_2.metrics_history]
        r2_history_2 = r2_history_2[exclude_rounds:]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds_1, r2_history_1, marker='o', color='green', label=f'{num_clients_1} Clients')
        plt.plot(rounds_2, r2_history_2, marker='o', color='red', label=f'{num_clients_2} Clients')
        plt.title('Global Model Validation R² Score over Rounds')
        plt.xlabel('Round')
        plt.ylabel('R² Score')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'results/{details}r2_over_rounds.png')
        plt.show()

    # Plot Validation Accuracy
    if strategy_1.accuracy_history and strategy_2.accuracy_history:
        plt.figure(figsize=(10, 5))
        plt.plot(rounds_1, accuracy_history_1, marker='o', color='blue', label=f'{num_clients_1} Clients')
        plt.plot(rounds_2, accuracy_history_2, marker='o', color='cyan', label=f'{num_clients_2} Clients')
        plt.title(f'Global Model Validation Accuracy (±{TOLERANCE*100}%) over Rounds')
        plt.xlabel('Round')
        plt.ylabel(f'Accuracy (Proportion within ±{TOLERANCE*100}%)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'results/{details}accuracy_over_rounds.png')
        plt.show()
