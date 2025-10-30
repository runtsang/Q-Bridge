from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class HybridEstimatorQNN:
    """Quantum neural network that mirrors the classical hybrid model."""
    def __init__(self, use_quantum: bool = True, seed: int = 42):
        self.use_quantum = use_quantum
        self.observable = SparsePauliOp.from_list([("Y", 1)])
        if use_quantum:
            self.circuit = QuantumCircuit(1)
            self.input_param = self.circuit.parameter("theta_in")
            self.weight_param = self.circuit.parameter("theta_w")
            self.circuit.h(0)
            self.circuit.ry(self.input_param, 0)
            self.circuit.rx(self.weight_param, 0)
            self.circuit.measure_all()
            self.backend = AerSimulator()
            self.compiled = transpile(self.circuit, self.backend)
            self.estimator = StatevectorEstimator()
            self.estimator_qnn = QiskitEstimatorQNN(
                circuit=self.circuit,
                observables=self.observable,
                input_params=[self.input_param],
                weight_params=[self.weight_param],
                estimator=self.estimator,
            )
        else:
            # trivial linear estimator fallback
            self.estimator_qnn = QiskitEstimatorQNN(
                circuit=QuantumCircuit(1),
                observables=self.observable,
                input_params=[],
                weight_params=[],
                estimator=StatevectorEstimator(),
            )

    def __call__(self, inputs, weights=None):
        if self.use_quantum:
            return self.estimator_qnn(inputs, weights)
        else:
            return self.estimator_qnn(inputs)

def EstimatorQNN() -> HybridEstimatorQNN:
    """Return a HybridEstimatorQNN instance for the quantum path."""
    return HybridEstimatorQNN()
