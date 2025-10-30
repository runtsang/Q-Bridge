"""Hybrid quantum sampler network that combines a quantum kernel with a
parameterized sampler circuit.

This module implements a `HybridSamplerQNN` class that mirrors the
classical hybrid sampler but replaces the RBF kernel with a quantum
kernel evaluated via a small parameterized circuit.  The sampler
circuit itself is identical to the original `SamplerQNN` helper.
The design allows a user to perform kernel‑based feature extraction
on quantum hardware or a simulator and then feed the resulting
feature vector into a small variational sampler.

The public API matches the original `SamplerQNN` helper for
drop‑in compatibility:

```python
from SamplerQNN__gen083 import SamplerQNN
model = SamplerQNN()
```

"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as QSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.quantum_info import Statevector

class QuantumKernel:
    """
    A lightweight quantum kernel that encodes two input vectors into a
    2‑qubit state and returns the probability of measuring |00⟩,
    which is equivalent to the squared overlap with the reference state.
    """
    def __init__(self, depth: int = 2):
        self.depth = depth
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        # Parameter vector for two input samples
        x = ParameterVector("x", 2)
        y = ParameterVector("y", 2)
        # Encode first sample
        qc.ry(x[0], 0)
        qc.ry(x[1], 1)
        # Entangle
        qc.cx(0, 1)
        # Encode second sample (opposite sign to compute overlap)
        qc.ry(-y[0], 0)
        qc.ry(-y[1], 1)
        # Optional additional layers
        for _ in range(self.depth - 1):
            qc.ry(x[0], 0)
            qc.ry(x[1], 1)
            qc.cx(0, 1)
            qc.ry(-y[0], 0)
            qc.ry(-y[1], 1)
            qc.cx(0, 1)
        return qc

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the kernel value for two 2‑dimensional vectors.

        Parameters
        ----------
        x, y : np.ndarray
            1‑D arrays of length 2.

        Returns
        -------
        float
            Squared overlap |⟨x|y⟩|^2, computed via a statevector simulator.
        """
        param_dict = {"x": x, "y": y}
        sv = Statevector.from_instruction(self.circuit.bind_parameters(param_dict))
        return abs(sv[0]) ** 2

class HybridSamplerQNN:
    """
    Quantum sampler network that augments the classical sampler with a
    quantum kernel.  The kernel is evaluated on a quantum simulator and
    the resulting scalar is used as an additional feature for the sampler
    circuit.
    """
    def __init__(self, kernel_depth: int = 2, sampler_depth: int = 2):
        self.kernel = QuantumKernel(depth=kernel_depth)
        self.sampler_circuit = self._build_sampler_circuit(sampler_depth)
        # The Qiskit SamplerQNN expects separate input and weight parameters.
        self.sampler = QSamplerQNN(
            circuit=self.sampler_circuit,
            input_params=ParameterVector("input", 2),
            weight_params=ParameterVector("weight", 4),
            sampler=QSampler()
        )

    def _build_sampler_circuit(self, depth: int) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        # Input encoding
        inp = ParameterVector("input", 2)
        qc.ry(inp[0], 0)
        qc.ry(inp[1], 1)
        qc.cx(0, 1)
        # Parameterized rotation layers
        for _ in range(depth):
            w = ParameterVector("weight", 4)
            qc.ry(w[0], 0)
            qc.ry(w[1], 1)
            qc.cx(0, 1)
            qc.ry(w[2], 0)
            qc.ry(w[3], 1)
        return qc

    def kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the Gram matrix between two datasets using the quantum kernel."""
        m, n = X.shape[0], Y.shape[0]
        K = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                K[i, j] = self.kernel.kernel(X[i], Y[j])
        return K

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Run the sampler circuit for a single input vector.

        Parameters
        ----------
        x : np.ndarray
            Shape ``(2,)`` – the raw data point.

        Returns
        -------
        np.ndarray
            Probabilities over two measurement outcomes.
        """
        param_dict = {
            "input": x,
            "weight": np.random.rand(4)
        }
        result = self.sampler.run(self.sampler_circuit, param_dict)
        return result[0].probabilities  # first (and only) simulation result

def SamplerQNN() -> HybridSamplerQNN:
    """
    Factory that returns a fresh instance of the hybrid quantum sampler.
    The function signature matches the original ``SamplerQNN`` helper
    for backward compatibility.
    """
    return HybridSamplerQNN()
