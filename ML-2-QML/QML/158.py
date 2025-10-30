"""Quantum expectation head used by the hybrid network."""

import numpy as np
import qiskit
from qiskit import assemble, transpile
import torch

class QCNet:
    """Parameterised two‑qubit circuit executed on the Aer simulator."""
    def __init__(self, n_qubits: int = 2, shots: int = 100, shift: float = np.pi / 2):
        self.n_qubits = n_qubits
        self.shots = shots
        self.shift = shift
        self.backend = qiskit.Aer.get_backend("aer_simulator")
        # Build a single template circuit
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.theta = qiskit.circuit.Parameter("theta")
        for q in range(n_qubits):
            self.circuit.ry(self.theta, q)
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit for each parameter in params (batched)."""
        compiled = transpile(self.circuit, self.backend)
        results = []
        for p in np.atleast_1d(params):
            bound = {self.theta: p}
            qobj = assemble(compiled, parameter_binds=[bound], shots=self.shots)
            job = self.backend.run(qobj)
            counts = job.result().get_counts()
            # Expectation of Z on the first qubit
            exp = 0.0
            for state, cnt in counts.items():
                exp += cnt if state[0] == '0' else -cnt
            exp /= self.shots
            results.append(exp)
        return np.array(results)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Wrap the run method to accept torch tensors."""
        if inputs.ndim > 1:
            inputs = inputs.view(-1)
        params = inputs.detach().cpu().numpy()
        exp = self.run(params)
        return torch.tensor(exp, dtype=torch.float32, device=inputs.device)

__all__ = ["QCNet"]
