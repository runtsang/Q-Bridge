from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator, Sampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
import numpy as np

class HybridSamplerQNN:
    """
    Quantum implementation of the hybrid sampler.
    Builds a feature‑map, a convolution‑pooling ansatz, and a sampling
    circuit that can be executed on any Qiskit backend.
    """
    def __init__(self):
        # Feature map
        self.feature_map = ZFeatureMap(8)
        # Ansatz
        self.circuit = self._build_ansatz()
        # Primitives
        self.estimator = Estimator()
        self.sampler = Sampler()
        # QNNs
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I"*7, 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator
        )
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit.decompose(),
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            sampler=self.sampler
        )

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct the convolution‑pooling ansatz used in the QCNN example."""
        def conv_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi/2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            qc.cx(1, 0)
            qc.rz(np.pi/2, 0)
            return qc

        def conv_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=num_qubits*3)
            idx = 0
            for i in range(0, num_qubits, 2):
                sub = conv_circuit(params[idx:idx+3])
                qc.append(sub, [i, i+1])
                qc.barrier()
                idx += 3
            return qc

        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi/2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def pool_layer(sources, sinks, prefix):
            qc = QuantumCircuit(len(sources)+len(sinks))
            params = ParameterVector(prefix, length=len(sources)*3)
            idx = 0
            for s, t in zip(sources, sinks):
                sub = pool_circuit(params[idx:idx+3])
                qc.append(sub, [s, t])
                qc.barrier()
                idx += 3
            return qc

        # Assemble the full ansatz
        ansatz = QuantumCircuit(8)
        ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
        ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), range(8), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), range(4,8), inplace=True)
        ansatz.compose(pool_layer([0,1], [2,3], "p2"), range(4,8), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), range(6,8), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), range(6,8), inplace=True)
        return ansatz

    def run(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Execute the sampler on the given input data.
        Parameters
        ----------
        inputs : np.ndarray
            Input vector matching the feature‑map dimension (8).
        shots : int
            Number of measurement shots for sampling.
        Returns
        -------
        np.ndarray
            Normalised probability distribution over measurement outcomes.
        """
        # Bind input parameters
        param_dict = {p: val for p, val in zip(self.feature_map.parameters, inputs)}
        # Sample
        counts = self.sampler_qnn.run(input_params=param_dict, shots=shots)
        # Convert counts to probabilities
        total = sum(counts.values())
        probs = np.array([c/total for c in counts.values()])
        return probs

def SamplerQNN() -> HybridSamplerQNN:
    """
    Factory mirroring the original API.
    Returns an instance of the quantum hybrid sampler.
    """
    return HybridSamplerQNN()

__all__ = ["HybridSamplerQNN", "SamplerQNN"]
