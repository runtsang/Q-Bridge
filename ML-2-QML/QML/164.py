import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.primitives import Sampler as QiskitSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class SamplerQNNExtended:
    """Quantum sampler that mirrors the classical SamplerQNNExtended API.

    The circuit contains two qubits, a flexible number of variational layers and a
    parameter‑shift gradient estimator. It supports both state‑vector and shot‑based
    probability extraction and exposes a ``forward`` method that returns a
    probability vector compatible with the classical implementation.

    Parameters
    ----------
    n_qubits : int, default 2
        Number of qubits.
    n_layers : int, default 3
        Number of entangling‑plus‑rotation layers.
    backend : str, default'statevector_simulator'
        Backend for probability extraction.
    shots : int, default None
        Number of shots for sampling; if None, use state‑vector probabilities.
    """
    def __init__(self,
                 n_qubits: int = 2,
                 n_layers: int = 3,
                 backend: str ='statevector_simulator',
                 shots: int | None = None) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend
        self.shots = shots

        # Parameters
        self.input_params = ParameterVector("x", n_qubits)
        self.weight_params = ParameterVector("w", n_qubits * n_layers * 2)

        # Build circuit
        self.circuit = self._build_circuit()

        # Primitive for sampling
        self.sampler = QiskitSampler(backend=self.backend)

        # Wrap into Qiskit ML SamplerQNN for convenience
        self.qnn = SamplerQNN(circuit=self.circuit,
                              input_params=self.input_params,
                              weight_params=self.weight_params,
                              sampler=self.sampler)

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits)
        qc = QuantumCircuit(qr)
        # Input rotations
        for i in range(self.n_qubits):
            qc.ry(self.input_params[i], qr[i])
        # Variational layers
        idx = 0
        for _ in range(self.n_layers):
            for i in range(self.n_qubits):
                qc.ry(self.weight_params[idx], qr[i]); idx += 1
            for i in range(self.n_qubits - 1):
                qc.cx(qr[i], qr[i + 1])
            for i in range(self.n_qubits):
                qc.ry(self.weight_params[idx], qr[i]); idx += 1
            # Entangle again
            for i in range(self.n_qubits - 1):
                qc.cx(qr[i + 1], qr[i])
        return qc

    def forward(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute probabilities from the variational circuit.

        Parameters
        ----------
        inputs : np.ndarray of shape (B, n_qubits)
            Input parameters for the Ry gates.
        weights : np.ndarray of shape (B, n_qubits * n_layers * 2)
            Variational parameters for the circuit.

        Returns
        -------
        probs : np.ndarray of shape (B, 2)
            Probability of measuring the computational basis states |00> and |01>.
            The returned vector is softmax‑normalized to sum to one.
        """
        # Validate shapes
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        if weights.ndim == 1:
            weights = weights[None, :]

        probs_list = []
        for inp, w in zip(inputs, weights):
            param_dict = {p: val for p, val in zip(self.input_params, inp)}
            param_dict.update({p: val for p, val in zip(self.weight_params, w)})
            bound = self.qnn.bind_parameters(param_dict)
            if self.shots is None:
                # Exact state‑vector probabilities
                state = Statevector.from_instruction(bound)
                probs = np.abs(state.data)**2
                probs = probs[[0, 1]]  # indices for |00> and |01>
                probs = probs / probs.sum()
            else:
                result = self.sampler.run(bound, shots=self.shots).result()
                counts = result.get_counts()
                probs = np.array([counts.get('00', 0), counts.get('01', 0)], dtype=float)
                probs /= probs.sum()
            probs_list.append(probs)
        return np.stack(probs_list)

    def sample(self, inputs: np.ndarray, weights: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Draw categorical samples from the quantum circuit output.

        Parameters
        ----------
        inputs : np.ndarray
            Input parameters.
        weights : np.ndarray
            Variational parameters.
        n_samples : int
            Number of draws per input.

        Returns
        -------
        samples : np.ndarray of shape (B, n_samples)
            Class indices 0 or 1.
        """
        probs = self.forward(inputs, weights)
        return np.random.choice(2, size=probs.shape[0] * n_samples, p=probs.ravel()).reshape(probs.shape[0], n_samples)

__all__ = ["SamplerQNNExtended"]
