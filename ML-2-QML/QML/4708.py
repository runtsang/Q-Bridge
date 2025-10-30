import numpy as np
import torch
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, param_values: np.ndarray) -> QuantumCircuit:
        if len(param_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch.")
        binding = dict(zip(self.parameters, param_values))
        return self.circuit.assign_parameters(binding, inplace=False)

    def evaluate(self, observables: list[SparsePauliOp], parameter_sets: list[np.ndarray]) -> list[list[complex]]:
        results = []
        for params in parameter_sets:
            circ = self._bind(params)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to deterministic estimator."""
    def evaluate(self, observables: list[SparsePauliOp], parameter_sets: list[np.ndarray],
                 shots: int | None = None, seed: int | None = None) -> list[list[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return [[float(v) for v in row] for row in raw]
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [rng.normal(float(v), max(1e-6, 1/np.sqrt(shots))) for v in row]
            noisy.append(noisy_row)
        return noisy

def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, list, list, list[SparsePauliOp]]:
    """Construct a layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("θ", num_qubits * depth)
    qc = QuantumCircuit(num_qubits)
    for i, qubit in enumerate(range(num_qubits)):
        qc.rx(encoding[i], qubit)
    idx = num_qubits
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

class HybridClassifier:
    """Quantum classifier that evaluates a parameterised circuit and returns expectation values."""
    def __init__(self, num_qubits: int = 4, depth: int = 2, shots: int = 1024,
                 shift: float = np.pi/2):
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)
        self.backend = Aer.get_backend("aer_simulator")
        self.shots = shots
        self.shift = shift

    def _run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit for the given parameters and return expectation of Z."""
        if params.ndim == 1:
            params = params[np.newaxis, :]
        bound = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, params)), inplace=False)
        compiled = transpile(bound, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        exp = 0.0
        for bitstring, count in counts.items():
            outcome = 1 if bitstring.count('1') % 2 == 0 else -1
            exp += outcome * count
        return np.array([exp / self.shots])

    def evaluate(self, parameter_sets: list[np.ndarray],
                 observables: list[SparsePauliOp],
                 shots: int | None = None, seed: int | None = None) -> list[list[float]]:
        """Compute expectations for each parameter set and observable, optionally adding shot noise."""
        estimator = FastEstimator(self.circuit) if shots else FastBaseEstimator(self.circuit)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return probabilities for binary classification from quantum expectations."""
        probs = []
        for i in range(inputs.shape[0]):
            params = inputs[i].detach().cpu().numpy()
            exp = self._run(params).item()
            probs.append(exp)
        probs = np.array(probs)
        probs = torch.tensor(probs, dtype=inputs.dtype, device=inputs.device)
        probs = torch.clamp(probs, 0, 1)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["HybridClassifier", "build_classifier_circuit", "FastEstimator", "FastBaseEstimator"]
