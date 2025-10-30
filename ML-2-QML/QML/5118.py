import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

class HybridQCNNAutoEncoder:
    """
    Quantum counterpart that builds a QCNN‑style circuit and exposes it
    as a SamplerQNN. The circuit consists of a feature map, several
    parameterized convolutional layers, pooling layers and a final
    ansatz.  The sampler QNN can be trained with a variational
    optimizer to perform classification.
    """
    def __init__(self,
                 num_qubits: int = 8,
                 conv_layers: int = 3,
                 pool_layers: int = 3,
                 feature_map_depth: int = 1,
                 ansatz_reps: int = 5) -> None:
        self.num_qubits = num_qubits
        self.conv_layers = conv_layers
        self.pool_layers = pool_layers
        self.feature_map_depth = feature_map_depth
        self.ansatz_reps = ansatz_reps
        self.circuit = self._build_circuit()

    def _conv_layer(self, qc: QuantumCircuit, layer_idx: int) -> QuantumCircuit:
        params = ParameterVector(f"c{layer_idx}_", length=self.num_qubits)
        for i in range(0, self.num_qubits, 2):
            qc.rz(params[i], i)
            qc.ry(params[i+1], i+1)
            qc.cx(i, i+1)
        return qc

    def _pool_layer(self, qc: QuantumCircuit, layer_idx: int) -> QuantumCircuit:
        params = ParameterVector(f"p{layer_idx}_", length=self.num_qubits//2)
        for i in range(0, self.num_qubits//2):
            qc.rz(params[i], 2*i)
            qc.ry(params[i], 2*i+1)
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        algorithm_globals.random_seed = 12345
        feature_map = ZFeatureMap(self.num_qubits, reps=self.feature_map_depth)
        ansatz = RealAmplitudes(self.num_qubits, reps=self.ansatz_reps)
        qc = QuantumCircuit(self.num_qubits)
        qc.compose(feature_map, inplace=True)
        for i in range(self.conv_layers):
            qc = self._conv_layer(qc, i)
        for i in range(self.pool_layers):
            qc = self._pool_layer(qc, i)
        qc.compose(ansatz, inplace=True)
        return qc

    def build_qnn(self) -> SamplerQNN:
        sampler = StatevectorSampler()
        observable = [("Z" + "I" * (self.num_qubits - 1), 1)]
        qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            sampler=sampler,
            interpret=lambda x: x
        )
        return qnn
