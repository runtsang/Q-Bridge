import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class QuantumClassifierModel:
    """
    Classical feed‑forward classifier that mimics the quantum helper interface.
    Extends the seed by adding L2 regularisation, early‑stopping,
    and a flexible hidden‑layer size.
    """
    def __init__(self, num_features: int, depth: int = 3,
                 hidden_dim: int = 64, lr: float = 1e-3,
                 weight_decay: float = 1e-4, device: str = "cpu"):
        self.device = torch.device(device)
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))  # binary classification
        self.net = nn.Sequential(*layers).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr,
                                    weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50,
            batch_size: int = 64, verbose: bool = True,
            patience: int = 5) -> None:
        """
        Train the network with early stopping on the training set itself.
        """
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                                torch.tensor(y, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        best_loss = np.inf
        epochs_no_improve = 0
        for epoch in range(epochs):
            self.net.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(xb)
                loss = self.criterion(outputs, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataset)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")
            # early‑stopping
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping after {epoch+1} epochs")
                    break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return class predictions (0 or 1) for the input data.
        """
        self.net.eval()
        with torch.no_grad():
            outputs = self.net(torch.tensor(X, dtype=torch.float32).to(self.device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return classification accuracy on the given dataset.
        """
        preds = self.predict(X)
        return np.mean(preds == y)

    def get_params(self) -> dict:
        """
        Return a dictionary of the network parameters.
        """
        return {k: v.cpu().numpy() for k, v in self.net.state_dict().items()}

    def set_params(self, params: dict) -> None:
        """
        Load the given parameters into the network.
        """
        state = {k: torch.tensor(v) for k, v in params.items()}
        self.net.load_state_dict(state)

__all__ = ["QuantumClassifierModel"]
