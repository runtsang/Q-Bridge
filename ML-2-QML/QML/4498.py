import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector, Parameter
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2

class SelfAttention__gen181Quantum:
    """
    Quantum implementation mirroring the hybrid classical model.
    Combines a parameterised quantum self‑attention circuit,
    a sampler QNN for classification, an estimator QNN for regression,
    and a small quantum feature encoder.
    """

    def __init__(self, n_qubits: int = 4, seed: np.random.Generator = np.random.default_rng()):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")

        # Quantum feature encoder (EfficientSU2)
        self.encoder = EfficientSU2(num_qubits=n_qubits, reps=2)

        # Self‑attention circuit
        self.attn_params = ParameterVector("attn", 3 * n_qubits)
        self.entangle_params = ParameterVector("ent", n_qubits - 1)
        self.attn_circuit = self._build_attn_circuit()

        # Sampler QNN
        self.sampler_qnn = self._build_sampler_qnn()

        # Estimator QNN
        self.estimator_qnn = self._build_estimator_qnn()

    def _build_attn_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(self.attn_params[3 * i], i)
            qc.ry(self.attn_params[3 * i + 1], i)
            qc.rz(self.attn_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(self.entangle_params[i], i, i + 1)
        return qc

    def _build_sampler_qnn(self) -> SamplerQNN:
        inputs = ParameterVector("in", 2)
        weights = ParameterVector("w", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        sampler = StatevectorSampler()
        return SamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)

    def _build_estimator_qnn(self) -> EstimatorQNN:
        inp = Parameter("inp")
        w = Parameter("w")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(inp, 0)
        qc.rx(w, 0)
        obs = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        return EstimatorQNN(circuit=qc, observables=obs,
                            input_params=[inp], weight_params=[w],
                            estimator=estimator)

    def encode_features(self, data: np.ndarray) -> dict:
        """
        Run the encoder on classical data reshaped to qubit states.
        """
        bound = {param: val for param, val in zip(self.encoder.parameters, data)}
        qc_bound = self.encoder.bind_parameters(bound)
        job = execute(qc_bound, self.backend, shots=1024)
        return job.result().get_counts(qc_bound)

    def run_attention(self, rotation_vals: np.ndarray, entangle_vals: np.ndarray) -> dict:
        bound = {p: v for p, v in zip(self.attn_params, rotation_vals)}
        bound.update({p: v for p, v in zip(self.entangle_params, entangle_vals)})
        qc = self.attn_circuit.bind_parameters(bound)
        job = execute(qc, self.backend, shots=1024)
        return job.result().get_counts(qc)

    def classify(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return self.sampler_qnn.run(inputs, weights)

    def regress(self, inputs: np.ndarray, weight: np.ndarray) -> float:
        return self.estimator_qnn.run(inputs, weight)
