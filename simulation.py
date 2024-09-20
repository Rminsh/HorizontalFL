# simulation.py
import flwr as fl
import torch
import matplotlib.pyplot as plt
from server import weighted_average
from server import strategy
from client import FlowerClient
from client import set_random_seeds
from client import load_data_for_client
from client import MLP

# Server Config
num_clients = 10 # Total number of clients
config = fl.server.ServerConfig(num_rounds=20) # Total number of rounds

def client_fn(cid: str):
    cid_int = int(cid)
    
    set_random_seeds(42)  # Set random seeds for reproducibility

    # Load partitioned data for this client
    train_loader, val_loader, test_loader, input_dim = load_data_for_client(cid_int, num_clients)

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim).to(device)

    # Create a Flower client
    return FlowerClient(model, train_loader, val_loader, test_loader, device, cid_int).to_client()

# Start Flower server
if __name__ == "__main__":
    # Start the server
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,                 
        config=config,
        strategy=strategy,
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
