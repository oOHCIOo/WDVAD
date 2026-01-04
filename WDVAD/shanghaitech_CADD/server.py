from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics
import argparse
from model import Model_single
from typing import Tuple, Optional
import torch
import torch.nn as nn
from flwr.common import Parameters, ndarrays_to_parameters

parser = argparse.ArgumentParser(description='Federated Learning Client Configuration')

parser.add_argument('--rounds', type=int, default=5,
                        help="Total communication rounds with server (default: %(default)d)")

parser.add_argument('--sample_fraction', type=float, default=1.0,
                        help="Fraction of active clients per round (default: %(default).1f)")

parser.add_argument('--min_num_clients', type=int, default=2,
                    help="Minimum number of available clients required for sampling (default: 2)")

parser.add_argument('--strategy', type=str, default='fedavg', choices=['fedavg', 'fedprox', 'qffedavg'],
                        help="Aggregation strategy (default: %(default)s)")

parser.add_argument('--server_address', type=str, default='localhost:8080',
                    help='gRPC server address (default: localhost:8080)')


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Thist function averages teh `accuracy` metric sent by the clients in a `evaluate`
    stage (i.e. clients received the global model and evaluate it on their local
    validation sets)."""
    # Multiply accuracy of each client by number of examples used
    AUC = [num_examples * m["AUC"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Calculate the weighted average accuracy
    aggregated_auc = sum(AUC) / sum(examples)

   # Aggregate and return custom metric (weighted average)
    return {"AUC": aggregated_auc}
def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 5,  # Number of local epochs done by clients
        "server_round":server_round
    }
    return config



def main():

    args = parser.parse_args()

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        fraction_evaluate=args.sample_fraction,
        min_fit_clients=args.min_num_clients,
        min_evaluate_clients=args.min_num_clients,
        min_available_clients=args.min_num_clients,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn = fit_config,
        evaluate_metrics_aggregation_fn=weighted_average
    )

    # Start Flower server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        grpc_max_message_length = 736870912
    )


if __name__ == "__main__":
    main()
