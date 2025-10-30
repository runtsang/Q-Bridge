"""Combined quantum module that mirrors the classical FCLGen407 API.

Each sub‑module is a parameterised Qiskit circuit that emulates the behaviour of the
corresponding classical layer.  The top‑level :class:`QuantumFCLGen407` exposes a
uniform interface that can be swapped in place of :class:`FCLGen407` for end‑to‑end
experiments.  The implementation favours clarity over hardware efficiency, making
it suitable for simulation‑based research.

"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit import Parameter

__all__ = [
    "QuantumFCLGen407",
    "QuantumFullyConnected",
    "QuantumConvFilter",
    "QuantumTransformerBlock",
    "QuantumSelfAttention",
]


class QuantumFullyConnected:
    """Parameterised quantum circuit that serves as a fully‑connected layer."""

    def __init__(self, n_qubits: int, backend, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self._circuit = QuantumCircuit(n_qubits, n_qubits)
        self.theta = Parameter("theta")

        # Prepare a simple H‑R_y circuit
        self._circuit.h(range(n_qubits))
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure(range(n_qubits), range(n_qubits))

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for each theta value and return the mean number of |1>."""
        param_binds = [{self.theta: t} for t in thetas]
        job = execute(
            self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds
        )
        result = job.result().get_counts(self._circuit)
        expectation = sum(int(bit) * count for bit, count in result.items()) / self.shots
        return np.array([expectation])


class QuantumConvFilter:
    """Quantum analogue of the convolutional filter.

    The circuit applies an Rx rotation to each qubit, followed by a small random
    entangling circuit, and finally measures all qubits.  The output is the
    average probability of measuring |1>, which is used as the convolution
    activation.
    """

    def __init__(self, n_qubits: int, backend, shots: int = 1024, threshold: float = 127) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.threshold = threshold
        self._circuit = QuantumCircuit(n_qubits, n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(n_qubits)]

        for i in range(n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(n_qubits, 2)
        self._circuit.measure(range(n_qubits), range(n_qubits))

    def run(self, data: np.ndarray) -> np.ndarray:
        """Map each element of ``data`` to a rotation parameter and evaluate the circuit."""
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data_flat:
            bind = {theta: np.pi if val > self.threshold else 0 for theta, val in zip(self.theta, dat)}
            param_binds.append(bind)

        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        # Compute average number of |1> over all shots
        counts = sum(int(bit) * val for bit, val in result.items())
        return np.array([counts / (self.shots * self.n_qubits)])


class QuantumTransformerBlock:
    """Very small quantum transformer block that mimics attention via a chain of CNOTs."""

    def __init__(self, n_qubits: int, backend, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self._circuit = QuantumCircuit(n_qubits, n_qubits)
        self.params = [Parameter(f"theta{i}") for i in range(n_qubits)]

        # Apply a rotation to each qubit
        for i in range(n_qubits):
            self._circuit.ry(self.params[i], i)

        # Entangle neighbours (circular chain)
        for i in range(n_qubits - 1):
            self._circuit.cx(i, i + 1)
        self._circuit.cx(n_qubits - 1, 0)

        self._circuit.measure(range(n_qubits), range(n_qubits))

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the transformer circuit with the supplied rotation parameters."""
        param_binds = [{param: val for param, val in zip(self.params, params)}]
        job = execute(
            self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds
        )
        result = job.result().get_counts(self._circuit)

        # Return the probability of measuring the all‑zero string as a proxy for attention output
        zero_prob = result.get("0" * self.n_qubits, 0) / self.shots
        return np.array([zero_prob])


class QuantumSelfAttention:
    """Quantum self‑attention circuit mirroring the original Qiskit example."""

    def __init__(self, n_qubits: int, backend, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self._circuit = QuantumCircuit(self.qr, self.cr)

        # Parameterised rotations
        self.rotation_params = [Parameter(f"rot{i}") for i in range(n_qubits * 3)]
        for i in range(n_qubits):
            self._circuit.rx(self.rotation_params[3 * i], i)
            self._circuit.ry(self.rotation_params[3 * i + 1], i)
            self._circuit.rz(self.rotation_params[3 * i + 2], i)

        # Entangling rotations
        self.entangle_params = [Parameter(f"ent{i}") for i in range(n_qubits - 1)]
        for i in range(n_qubits - 1):
            self._circuit.crx(self.entangle_params[i], i, i + 1)

        self._circuit.measure(self.qr, self.cr)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = None,
    ) -> dict:
        """Execute the attention circuit and return the measurement counts."""
        shots = shots or self.shots
        param_binds = [
            {
                param: val
                for param, val in zip(self.rotation_params, rotation_params)
            }
        ] + [
            {
                param: val
                for param, val in zip(self.entangle_params, entangle_params)
            }
        ]

        job = execute(self._circuit, self.backend, shots=shots, parameter_binds=param_binds)
        return job.result().get_counts(self._circuit)


class QuantumFCLGen407:
    """Unified interface that bundles the quantum sub‑modules."""

    def __init__(
        self,
        n_fc_qubits: int = 1,
        n_conv_qubits: int = 4,
        n_transformer_qubits: int = 8,
        n_attention_qubits: int = 4,
        shots: int = 1024,
    ) -> None:
        backend = Aer.get_backend("qasm_simulator")
        self.fc = QuantumFullyConnected(n_fc_qubits, backend, shots)
        self.conv = QuantumConvFilter(n_conv_qubits, backend, shots)
        self.transformer = QuantumTransformerBlock(n_transformer_qubits, backend, shots)
        self.attention = QuantumSelfAttention(n_attention_qubits, backend, shots)

    def run_fc(self, thetas: np.ndarray) -> np.ndarray:
        return self.fc.run(thetas)

    def run_conv(self, data: np.ndarray) -> np.ndarray:
        return self.conv.run(data)

    def run_transformer(self, params: np.ndarray) -> np.ndarray:
        return self.transformer.run(params)

    def run_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = None,
    ) -> dict:
        return self.attention.run(rotation_params, entangle_params, shots)
