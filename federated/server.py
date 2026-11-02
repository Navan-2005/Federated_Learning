import random
from fastapi import FastAPI
from aggregator import fedavg
from evaluation import evaluate


class Server:
    def __init__(self, global_model, clients, X_test, y_test):
        self.global_model = global_model
        self.clients = clients
        self.X_test = X_test
        self.y_test = y_test
        self.history = []

    def run_round(self, round_num, local_epochs, lr, client_fraction=1.0):
        k = max(1, int(len(self.clients) * client_fraction))
        selected = random.sample(self.clients, k)

        updates, losses = [], []
        for client in selected:
            weights, n, loss = client.train(self.global_model, local_epochs, lr)
            updates.append((weights, n))
            losses.append(loss)

        self.global_model.load_state_dict(fedavg(updates))
        metrics = evaluate(self.global_model, self.X_test, self.y_test)
        metrics["round"] = round_num
        metrics["train_loss"] = sum(losses) / len(losses)
        self.history.append(metrics)
        return metrics

    def train(self, rounds, local_epochs=5, lr=0.01, client_fraction=1.0):
        for r in range(1, rounds + 1):
            m = self.run_round(r, local_epochs, lr, client_fraction)
            print(
                f"Round {r:02d} | loss {m['train_loss']:.4f} | acc {m['accuracy']:.4f} "
                f"| recall {m['recall']:.4f} | f1 {m['f1']:.4f}"
            )
        return self.history
