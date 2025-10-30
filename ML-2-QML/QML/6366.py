import qiskit
from qiskit import QuantumCircuit, Aer, assemble, transpile
import numpy as np
import torch

class SamplerQNN__gen179:
    """
    Quantum sampler that implements a parameterised two‑qubit circuit.
    The input tensor is interpreted as rotation angles for the first
    layer of Ry gates.  A second layer of Ry gates with fixed weights
    refines the distribution.  The measurement probability of the
    |00⟩ outcome is returned as a torch tensor.
    """

    def __init__(self, backend=None, shots=1024):
        self.backend = backend or Aer.get_backend('aer_simulator')
        self.shots = shots

        # Build a template circuit with two parameters
        self.theta = qiskit.circuit.ParameterVector("theta", 2)
        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.theta[0], 0)
        self.circuit.ry(self.theta[1], 1)
        self.circuit.cx(0, 1)
        # Fixed second layer
        self.circuit.ry(np.pi / 4, 0)
        self.circuit.ry(np.pi / 3, 1)
        self.circuit.measure_all()

    def _run(self, params: np.ndarray) -> np.ndarray:
        """Execute the parameterised circuit for a single set of angles."""
        bound = {self.theta[0]: params[0], self.theta[1]: params[1]}
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots, parameter_binds=[bound])
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        # Compute probability of |00⟩
        prob_00 = counts.get("00", 0) / self.shots
        return np.array([prob_00])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Accepts a batch of 2‑dimensional rotation angles and returns a
        probability distribution over the two classical outcomes
        (|00⟩ vs not |00⟩).
        """
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        probs = []
        for angles in inputs.cpu().numpy():
            probs.append(self._run(angles))
        probs = np.concatenate(probs, axis=0)
        probs = torch.tensor(probs, dtype=torch.float32, device=inputs.device)
        # Convert to a two‑component distribution
        probs = torch.cat([probs, 1 - probs], dim=-1)
        return probs

__all__ = ["SamplerQNN__gen179"]
