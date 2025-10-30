import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class HybridSelfAttentionEstimator:
    """
    Quantum hybrid architecture: variational self‑attention circuit + EstimatorQNN.
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("statevector_simulator")
        self.attention_circuit = self._build_attention_circuit()
        self.estimator = self._build_estimator()

    def _build_attention_circuit(self) -> QuantumCircuit:
        rot = Parameter("θ")
        ent = Parameter("ϕ")
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rot, i)
            qc.ry(rot, i)
            qc.rz(rot, i)
        for i in range(self.n_qubits - 1):
            qc.crx(ent, i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    def _build_estimator(self) -> EstimatorQNN:
        w = Parameter("w")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rx(w, 0)
        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        return EstimatorQNN(circuit=qc,
                            observables=observable,
                            input_params=[],
                            weight_params=[w],
                            estimator=estimator)

    def run(self, rotation: float, entangle: float, shots: int = 1024):
        circ = self.attention_circuit.bind_parameters({self.attention_circuit.parameters[0]: rotation,
                                                       self.attention_circuit.parameters[1]: entangle})
        job = qiskit.execute(circ, self.backend, shots=shots)
        raw_counts = job.result().get_counts(circ)
        return self.estimator.predict(raw_counts)

__all__ = ["HybridSelfAttentionEstimator"]
