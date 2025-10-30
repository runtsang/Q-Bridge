"""Hybrid quantum network: quantum quanvolution filter followed by quantum fully connected layer."""

import numpy as np
import qiskit

class QuantumQuanvolutionFilter:
    """Quantum implementation of a 2x2 patch encoder with a random layer."""
    def __init__(self, n_qubits=4, backend=None, shots=100):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

    def encode_patch(self, patch: np.ndarray) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        for i, val in enumerate(patch):
            qc.ry(val, i)
        return qc

    def random_layer(self, qc: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
        for i in range(self.n_qubits):
            qc.cx(i, (i + 1) % self.n_qubits)
        return qc

    def measure_expectation(self, qc: qiskit.QuantumCircuit) -> np.ndarray:
        qc.measure_all()
        job = qiskit.execute(qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(qc)
        probs = np.array([counts.get(bin(i)[2:].zfill(self.n_qubits), 0) for i in range(2**self.n_qubits)]) / self.shots
        states = np.array([int(bin(i)[2:].zfill(self.n_qubits)[::-1], 2) for i in range(2**self.n_qubits)])
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def run(self, patches: np.ndarray) -> np.ndarray:
        expectations = []
        for patch in patches:
            qc = self.encode_patch(patch)
            qc = self.random_layer(qc)
            exp = self.measure_expectation(qc)
            expectations.append(exp)
        return np.concatenate(expectations)

class QuantumFullyConnectedLayer:
    """Parameterised quantum linear mapping from inputs to outputs."""
    def __init__(self, n_inputs: int, n_outputs: int, backend=None, shots=100):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.theta = np.random.randn(n_outputs, n_inputs)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        logits = []
        for out in range(self.n_outputs):
            qc = qiskit.QuantumCircuit(self.n_inputs)
            for i, val in enumerate(inputs):
                qc.ry(val + self.theta[out, i], i)
            qc.measure_all()
            job = qiskit.execute(qc, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(qc)
            probs = np.array([counts.get(bin(i)[2:].zfill(self.n_inputs), 0) for i in range(2**self.n_inputs)]) / self.shots
            states = np.array([int(bin(i)[2:].zfill(self.n_inputs)[::-1], 2) for i in range(2**self.n_inputs)])
            expectation = np.sum(states * probs)
            logits.append(expectation)
        return np.array(logits)

class SharedClassName:
    """Hybrid quantum network: quantum quanvolution filter followed by quantum fully connected layer."""
    def __init__(self, n_qubits_per_patch=4, n_fc_outputs=10, backend=None, shots=100):
        self.filter = QuantumQuanvolutionFilter(n_qubits=n_qubits_per_patch, backend=backend, shots=shots)
        self.fc = QuantumFullyConnectedLayer(n_inputs=4 * 14 * 14, n_outputs=n_fc_outputs, backend=backend, shots=shots)

    def run(self, x: np.ndarray) -> np.ndarray:
        # x shape (batch, 1, 28, 28)
        batch = x.shape[0]
        all_logits = []
        for i in range(batch):
            image = x[i, 0]
            patches = []
            for r in range(0, 28, 2):
                for c in range(0, 28, 2):
                    patch = image[r:r+2, c:c+2].flatten()
                    patches.append(patch)
            patches = np.array(patches)
            conv_out = self.filter.run(patches)  # shape (num_patches, 1)
            conv_out = conv_out.reshape(-1)
            logits = self.fc.run(conv_out)
            all_logits.append(logits)
        return np.stack(all_logits)

__all__ = ["SharedClassName"]
