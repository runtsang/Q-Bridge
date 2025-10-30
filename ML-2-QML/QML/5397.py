import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Pauli
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QuantumFraudFeatureExtractor:
    """
    Parameterised 2‑qubit circuit that outputs expectation values of Z on each qubit.
    Parameters are supplied at runtime, allowing the extractor to serve as a feature
    vector generator for the hybrid classical model.
    """
    def __init__(self, n_qubits: int = 2):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits)
        self.cr = ClassicalRegister(n_qubits)
        self.circuit = QuantumCircuit(self.qr, self.cr)
        self._build_circuit()
        self.estimator = StatevectorEstimator()

    def _build_circuit(self):
        # single‑qubit rotations
        for i in range(self.n_qubits):
            theta = Parameter(f"theta_{i}")
            phi = Parameter(f"phi_{i}")
            self.circuit.ry(theta, i)
            self.circuit.rz(phi, i)
        # entanglement
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)

    def evaluate(self, parameter_sets: np.ndarray) -> np.ndarray:
        """
        Return expectation values of Pauli‑Z on each qubit for each parameter set.
        """
        results = []
        for params in parameter_sets:
            mapping = dict(zip(self.circuit.parameters, params))
            bound_circ = self.circuit.assign_parameters(mapping, inplace=False)
            state = Statevector.from_instruction(bound_circ)
            exps = [state.expectation_value(Pauli("Z")) for _ in range(self.n_qubits)]
            results.append(exps)
        return np.array(results)

class QuantumSelfAttention:
    """
    Minimal quantum self‑attention circuit that mirrors the classical SelfAttention
    class.  The circuit is parameterised by rotation and entanglement angles.
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits)
        self.cr = ClassicalRegister(n_qubits)
        self.circuit = QuantumCircuit(self.qr, self.cr)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        circ.measure(self.qr, self.cr)
        return circ

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circ = self._build_circuit(rotation_params, entangle_params)
        backend = Aer.get_backend("qasm_simulator")
        job = execute(circ, backend, shots=shots)
        return job.result().get_counts(circ)

class QuantumEstimatorQNN:
    """
    Wrapper around Qiskit Machine Learning's EstimatorQNN for a 1‑qubit circuit.
    """
    def __init__(self):
        # Define a simple parameterised circuit
        self.params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(self.params[0], 0)
        qc.rx(self.params[1], 0)

        # Observable: Pauli‑Y
        observable = Pauli("Y")

        # Estimator
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[self.params[0]],
            weight_params=[self.params[1]],
            estimator=estimator,
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the quantum neural network on a batch of inputs.
        """
        return self.estimator_qnn.predict(inputs)

class FastEstimator:
    """
    Extends the deterministic FastBaseEstimator with Gaussian shot‑noise.
    """
    def __init__(self, circuit: QuantumCircuit, shots: int | None = None, seed: int | None = None):
        self.circuit = circuit
        self.shots = shots
        self.seed = seed
        self.backend = Aer.get_backend("statevector_simulator")

    def evaluate(self, observables: list[Pauli], parameter_sets: np.ndarray) -> np.ndarray:
        results = []
        for params in parameter_sets:
            bound_circ = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, params)), inplace=False)
            state = Statevector.from_instruction(bound_circ)
            exps = [state.expectation_value(obs) for obs in observables]
            if self.shots is not None:
                rng = np.random.default_rng(self.seed)
                exps = [float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in exps]
            results.append(exps)
        return np.array(results)

__all__ = [
    "QuantumFraudFeatureExtractor",
    "QuantumSelfAttention",
    "QuantumEstimatorQNN",
    "FastEstimator",
]
