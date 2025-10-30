import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector

class QuantumQuanvolution:
    """Pure‑quantum quanvolution circuit."""
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        # Base circuit template
        self.base_circuit = QuantumCircuit(n_qubits)
        self.theta = Parameter("θ")
        self.base_circuit.h(range(n_qubits))
        self.base_circuit.barrier()
        self.base_circuit.ry(self.theta, range(n_qubits))
        self.base_circuit.measure_all()

    def _build_circuit(self, pixel_values: np.ndarray) -> QuantumCircuit:
        """Create a circuit for a single 2×2 patch."""
        circ = self.base_circuit.copy()
        # Bind each pixel to a Ry rotation
        for i, val in enumerate(pixel_values):
            circ.ry(val, i)
        return circ

    def run(self, images: np.ndarray) -> np.ndarray:
        """
        images: (batch, 28, 28) grayscale values in [0, π].
        Returns a feature matrix of shape (batch, 4*14*14).
        """
        batch = images.shape[0]
        features = []
        for img in images:
            patch_feats = []
            for r in range(0, 28, 2):
                for c in range(0, 28, 2):
                    patch = img[r:r+2, c:c+2].flatten()
                    circ = self._build_circuit(patch)
                    job = qiskit.execute(circ, self.backend, shots=self.shots)
                    result = job.result()
                    counts = result.get_counts(circ)
                    probs = np.array(list(counts.values())) / self.shots
                    states = np.array([int(s, 2) for s in counts.keys()])
                    expectation = np.sum(states * probs)
                    patch_feats.append(expectation)
            features.append(patch_feats)
        return np.array(features)

class FastQuantumEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, values: list) -> QuantumCircuit:
        if len(values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.parameters, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: list, parameter_sets: list) -> list:
        results = []
        for vals in parameter_sets:
            circ = self._bind(vals)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["QuantumQuanvolution", "FastQuantumEstimator"]
