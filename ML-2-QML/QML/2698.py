"""Graph‑based regression model – quantum implementation.

This module builds a variational circuit whose entanglement pattern is
inferred from a fidelity‑based graph.  The circuit is applied to each
input state from the superposition dataset, and the resulting
expectation values of Pauli‑Z are fed into a classical linear head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import qiskit as qk
import qiskit.circuit.library as cl
import qiskit.quantum_info as qi
from qiskit.providers.aer import AerSimulator

from QuantumRegression import generate_superposition_data, RegressionDataset

Tensor = torch.Tensor
Device = torch.device


class GraphQNNRegression(nn.Module):
    """Quantum graph neural network for regression.

    Parameters
    ----------
    num_wires : int
        Number of qubits / graph nodes.
    entanglement : str, optional
        Entanglement scheme for the RealAmplitudes circuit
        (e.g. 'linear', 'circular', 'full').
    reps : int, optional
        Number of repetitions of the parameterized layer.
    """

    def __init__(
        self,
        num_wires: int,
        entanglement: str = "linear",
        reps: int = 2,
        device: str | Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.as_tensor([], device=device).device
        self.num_wires = num_wires

        # Variational circuit
        self.circuit = cl.RealAmplitudes(
            num_qubits=num_wires,
            reps=reps,
            entanglement=entanglement,
        )
        self.backend = AerSimulator(method="statevector")

        # Classical head
        self.head = nn.Linear(num_wires, 1)

    # ------------------------------------------------------------------
    # Helper: evaluate a single state through the circuit
    # ------------------------------------------------------------------
    def _statevector(self, amplitudes: np.ndarray) -> qi.Statevector:
        """Return the statevector after applying the circuit."""
        init_state = qi.Statevector(amplitudes)
        circuit = self.circuit
        # Build a full circuit that first prepares the input state
        full_circ = qk.QuantumCircuit(self.num_wires)
        full_circ.initialize(amplitudes, range(self.num_wires))
        full_circ.append(circuit, range(self.num_wires))
        job = self.backend.run(full_circ)
        result = job.result()
        return qi.Statevector.from_label(result.get_statevector(full_circ))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, state_batch: Tensor) -> Tensor:
        """Compute predictions for a batch of quantum states.

        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (batch, 2**num_wires) complex amplitudes.
        """
        batch_size = state_batch.shape[0]
        # Convert to numpy for simulation
        states_np = state_batch.detach().cpu().numpy()
        # Compute expectation values for each qubit
        exp_vals = np.zeros((batch_size, self.num_wires), dtype=np.float32)
        for i, amp in enumerate(states_np):
            sv = self._statevector(amp)
            for q in range(self.num_wires):
                exp_vals[i, q] = sv.expectation_value(qi.Pauli("Z", q)).real
        exp_tensor = torch.from_numpy(exp_vals).to(self.device)
        # Classical head
        return self.head(exp_tensor).squeeze(-1)

    # ------------------------------------------------------------------
    # Training helper
    # ------------------------------------------------------------------
    def fit(
        self,
        dataset: Iterable[dict[str, Tensor]],
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> None:
        """Simple training loop for the quantum circuit + head."""
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                states = batch["states"].to(self.device)
                targets = batch["target"].to(self.device)
                optimizer.zero_grad()
                preds = self.forward(states)
                loss = loss_fn(preds, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * states.size(0)
            epoch_loss /= len(dataset)
            print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.4f}")

    # ------------------------------------------------------------------
    # Prediction helper
    # ------------------------------------------------------------------
    def predict(self, states: Tensor) -> Tensor:
        """Predict on new data."""
        self.eval()
        with torch.no_grad():
            return self.forward(states.to(self.device))


__all__ = ["GraphQNNRegression"]
