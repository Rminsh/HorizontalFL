# simulation.py
import flwr as fl
import torch
import matplotlib.pyplot as plt
from server import weighted_average, CustomFedAvg
from client import FlowerClient
from client import set_random_seeds
from client import load_data_for_client
from client import MLP
from client import LinearRegressionModel

# Server Config
num_clients_1 = 3
num_clients_2 = 5
num_rounds = 70
details = str(num_rounds) + 'rounds_'

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
    return FlowerClient(model, train_loader, val_loader, test_loader, device, cid_int).to_client()

# Start Flower server
if __name__ == "__main__":
    # Start the server
    hist = fl.simulation.start_simulation(
        client_fn=client_fn_1,
        num_clients=num_clients_1,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy_1,
    )

    hist = fl.simulation.start_simulation(
        client_fn=client_fn_2,
        num_clients=num_clients_2,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy_2,
    )

    # Define the number of rounds to exclude
    exclude_rounds = 0  # Number of initial rounds to exclude

    # Ensure there are enough rounds to exclude
    if len(strategy_1.loss_history) <= exclude_rounds:
        raise ValueError(f"Not enough rounds to exclude the first {exclude_rounds} rounds.")

    # Define the rounds and corresponding loss/history excluding the first two rounds
    rounds_1 = range(exclude_rounds + 1, len(strategy_1.loss_history) + 1)
    loss_history_1 = strategy_1.loss_history[exclude_rounds:]
    rounds_2 = range(exclude_rounds + 1, len(strategy_2.loss_history) + 1)
    loss_history_2 = strategy_2.loss_history[exclude_rounds:]
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_1, loss_history_1, marker='o', label='10 Clients')
    plt.plot(rounds_2, loss_history_2, marker='o', label='20 Clients')
    plt.title('Global Model Loss over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/' + details + 'loss_over_rounds.png')
    plt.show()

    if strategy_1.metrics_history and 'accuracy' in strategy_1.metrics_history[0]:
        accuracy_history_1 = [m['accuracy'] for m in strategy_1.metrics_history]
        accuracy_history_1 = accuracy_history_1[exclude_rounds:]
        accuracy_history_2 = [m['accuracy'] for m in strategy_2.metrics_history]
        accuracy_history = accuracy_history_1[exclude_rounds:]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds_1, accuracy_history_1, marker='o', color='orange', label='10 Clients')
        plt.plot(rounds_2, accuracy_history_2, marker='o', color='indigo', label='20 Clients')
        plt.title('Accuracy over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/' + details + 'accuracy_over_rounds.png')
        plt.show()

    # Plot Validation MAE
    if strategy_1.metrics_history and 'mae' in strategy_1.metrics_history[0]:
        mae_history_1 = [m['mae'] for m in strategy_1.metrics_history]
        mae_history_1 = mae_history_1[exclude_rounds:]
        mae_history_2 = [m['mae'] for m in strategy_2.metrics_history]
        mae_history_2 = mae_history_2[exclude_rounds:]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds_1, mae_history_1, marker='o', color='orange', label='10 Clients')
        plt.plot(rounds_2, mae_history_2, marker='o', color='indigo', label='20 Clients')
        plt.title('Global Model Validation MAE over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Mean Absolute Error')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/' + details + 'mae_over_rounds.png')
        plt.show()

    # Plot Validation R² Score
    if strategy_1.metrics_history and 'r2' in strategy_1.metrics_history[0]:
        r2_history_1 = [m['r2'] for m in strategy_1.metrics_history]
        r2_history_1 = r2_history_1[exclude_rounds:]
        r2_history_2 = [m['r2'] for m in strategy_2.metrics_history]
        r2_history_2 = r2_history_2[exclude_rounds:]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds_1, r2_history_1, marker='o', color='green', label='10 Clients')
        plt.plot(rounds_2, r2_history_2, marker='o', color='red', label='20 Clients')
        plt.title('Global Model Validation R² Score over Rounds')
        plt.xlabel('Round')
        plt.ylabel('R² Score')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/' + details + 'r2_over_rounds.png')
        plt.show()

    # # Plot Validation MSE (from fit_metrics_history)
    # if strategy.fit_metrics_history and 'val_mse' in strategy.fit_metrics_history[0]:
    #     val_mse_history = [m['val_mse'] for m in strategy.fit_metrics_history]
    #     val_mse_history = val_mse_history[exclude_rounds:]
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(rounds, val_mse_history, marker='o', color='purple', label='Validation MSE')
    #     plt.title('Global Model Validation MSE over Rounds')
    #     plt.xlabel('Round')
    #     plt.ylabel('Validation MSE')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig('results/' + details + 'val_mse_over_rounds.png')
    #     plt.show()
