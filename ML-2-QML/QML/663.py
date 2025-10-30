import pennylane as qml
import numpy as np
import torch
from typing import Sequence, Dict, Any, Optional

class QuantumKernelMethod:
    """
    Quantum kernel implementation using PennyLane.
    Encodes data into a quantum state via a parameterized ansatz and
    computes the fidelity between states as the kernel value.
    Provides GPU support, state caching, and a simple grid search over
    the number of qubits.
    """
    def __init__(self,
                 n_wires: int = 4,
                 device_name: str = 'default.qubit',
                 use_gpu: bool = False,
                 cache: bool = True):
        self.n_wires = n_wires
        self.device_name = device_name
        self.cache = cache
        self._state_cache: Dict[tuple, torch.Tensor] = {}
        # Create PennyLane device; interface='torch' for GPU support
        self.device = qml.device(device_name, wires=n_wires)
        # Define a simple ansatz that encodes data into rotation angles
        @qml.qnode(self.device, interface='torch')
        def circuit(x):
            for i in range(n_wires):
                qml.RY(x[i], wires=i)
            return qml.state()
        self.circuit = circuit

    def _state_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the state vector for a batch of inputs.
        """
        key = (tuple(x.shape),)
        if self.cache and key in self._state_cache:
            return self._state_cache[key]
        # Ensure input is torch tensor
        x = x.to(torch.float32)
        # Compute state vectors for each sample
        states = []
        for sample in x:
            states.append(self.circuit(sample))
        states = torch.stack(states)  # shape (n_samples, 2**n_wires)
        if self.cache:
            self._state_cache[key] = states
        return states

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel matrix between x and y.
        """
        psi_x = self._state_vector(x)  # (n_x, 2**n_wires)
        psi_y = self._state_vector(y)  # (n_y, 2**n_wires)
        # Fidelity: |<psi_x|psi_y>|^2
        K = torch.abs(torch.matmul(psi_x, psi_y.t())) ** 2
        return K

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor],
                      n_wires: int = 4,
                      device_name: str = 'default.qubit',
                      cache: bool = True) -> np.ndarray:
        """
        Compute the Gram matrix between two sequences of tensors using the quantum kernel.
        """
        # Flatten tensors
        a_flat = [t.reshape(t.shape[0], -1).cpu().numpy() for t in a]
        b_flat = [t.reshape(t.shape[0], -1).cpu().numpy() for t in b]
        a_torch = torch.tensor(np.vstack(a_flat))
        b_torch = torch.tensor(np.vstack(b_flat))
        model = QuantumKernelMethod(n_wires=n_wires,
                                    device_name=device_name,
                                    cache=cache)
        K = model.forward(a_torch, b_torch)
        return K.cpu().numpy()

    def grid_search(self,
                    X: torch.Tensor,
                    n_wires_list: Sequence[int]) -> int:
        """
        Grid search over the number of qubits to maximize the average fidelity.
        Returns the best n_wires.
        """
        best_n = None
        best_score = -np.inf
        for n in n_wires_list:
            # Recreate device and circuit with new number of wires
            self.n_wires = n
            self.device = qml.device(self.device_name, wires=n)
            @qml.qnode(self.device, interface='torch')
            def circuit(x):
                for i in range(n):
                    qml.RY(x[i], wires=i)
                return qml.state()
            self.circuit = circuit
            self._state_cache.clear()
            K = self.forward(X, X)
            score = torch.mean(K).item()
            if score > best_score:
                best_score = score
                best_n = n
        self.n_wires = best_n
        # Recreate final device and circuit
        self.device = qml.device(self.device_name, wires=best_n)
        @qml.qnode(self.device, interface='torch')
        def circuit(x):
            for i in range(best_n):
                qml.RY(x[i], wires=i)
            return qml.state()
        self.circuit = circuit
        return best_n

__all__ = ['QuantumKernelMethod']
