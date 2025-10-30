"""Quantum sampler network mirroring SamplerQNNGen."""
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
import numpy as np

class SamplerQNNGen:
    """Quantum counterpart to the hybrid classical SamplerQNNGen."""
    def __init__(self, latent_dim: int = 32, num_qubits: int = 8):
        self.latent_dim = latent_dim
        self.num_qubits = num_qubits
        self.sampler = Sampler()
        self.qnn = self._build_qnn()

    # ------------------------------------------------------------------
    # QCNN‑style convolution and pooling blocks
    # ------------------------------------------------------------------
    def _conv_block(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(f"{prefix}_c", length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            layer = self._conv_layer(params[i * 3 : (i + 2) * 3])
            qc.append(layer.to_instruction(), [i, i + 1])
        return qc

    def _pool_block(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(f"{prefix}_p", length=(num_qubits // 2) * 3)
        for i in range(0, num_qubits, 2):
            layer = self._pool_layer(params[(i // 2) * 3 : ((i // 2) + 1) * 3])
            qc.append(layer.to_instruction(), [i, i + 1])
        return qc

    def _conv_layer(self, params) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _pool_layer(self, params) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # ------------------------------------------------------------------
    # Build the full QNN circuit
    # ------------------------------------------------------------------
    def _build_qnn(self) -> SamplerQNN:
        # Feature map
        feature_map = ZFeatureMap(self.num_qubits)

        # Ansatz with QCNN blocks
        ansatz = QuantumCircuit(self.num_qubits)
        ansatz.compose(self._conv_block(self.num_qubits, "c1"), in_bits=range(self.num_qubits), inplace=True)
        ansatz.compose(self._pool_block(self.num_qubits, "p1"), in_bits=range(self.num_qubits), inplace=True)
        # Second level
        ansatz.compose(self._conv_block(self.num_qubits // 2, "c2"), in_bits=range(self.num_qubits // 2), inplace=True)
        ansatz.compose(self._pool_block(self.num_qubits // 2, "p2"), in_bits=range(self.num_qubits // 2), inplace=True)
        # Third level
        ansatz.compose(self._conv_block(self.num_qubits // 4, "c3"), in_bits=range(self.num_qubits // 4), inplace=True)
        ansatz.compose(self._pool_block(self.num_qubits // 4, "p3"), in_bits=range(self.num_qubits // 4), inplace=True)

        # Autoencoder ansatz (RealAmplitudes)
        ae = RealAmplitudes(self.latent_dim, reps=5)
        ansatz.compose(ae.to_instruction(), range(self.latent_dim), inplace=True)

        # Full circuit
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(feature_map, range(self.num_qubits), inplace=True)
        circuit.compose(ansatz, range(self.num_qubits), inplace=True)

        # Observable for the SamplerQNN
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])

        # Wrap into a SamplerQNN
        qnn = SamplerQNN(
            circuit=circuit,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            sampler=self.sampler,
            output_shape=(2,),
            interpret=lambda x: x,
        )
        return qnn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """
        Sample from the quantum sampler.

        Parameters
        ----------
        inputs : np.ndarray
            Classical inputs for the feature map, shape (batch, num_qubits).

        Returns
        -------
        np.ndarray
            Sampled outcomes of shape (batch, 2).
        """
        return self.qnn.sample(inputs)

    def parameters(self):
        """Return the trainable weight parameters of the QNN."""
        return self.qnn.weight_params

__all__ = ["SamplerQNNGen"]
