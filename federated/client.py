import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Client:
    def __init__(self, client_id, X, y, batch_size=64):
        self.client_id = client_id
        self.num_samples = len(X)
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, global_model, epochs=5, lr=0.01, desc=None):
        """Train a copy of the global model locally; return (weights, num_samples, avg_loss).

        desc: label for a live tqdm progress bar (hidden when output is not a terminal).
        """
        model = copy.deepcopy(global_model)
        model.train()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        total_loss, steps = 0.0, 0
        bar = tqdm(total=epochs * len(self.loader), desc=desc, leave=False,
                   disable=None if desc else True, unit="batch")
        for epoch in range(1, epochs + 1):
            for xb, yb in self.loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
                bar.set_postfix(epoch=f"{epoch}/{epochs}", loss=f"{total_loss / steps:.4f}")
                bar.update(1)
        bar.close()
        return model.state_dict(), self.num_samples, total_loss / max(steps, 1)
