from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridSamplerQNN:
    """
    Quantum sampler that fuses the QCNN ansatz with the SamplerQNN input
    encoding.  The circuit consists of a 2‑qubit ZFeatureMap followed by
    three convolution‑like layers (each parameterised by three angles),
    mirroring the QCNN construction.  A StatevectorSampler is used as the
    backend, and the resulting SamplerQNN instance exposes a ``predict``
    method that returns a 2‑class probability distribution.
    """
    def __init__(self) -> None:
        # Parameters for data encoding and ansatz weights
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 12)  # 3 layers × 4 params
        self.circuit = self._build_circuit()
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        # Feature map – equivalent to the 2‑qubit ZFeatureMap in Quantum‑NAT
        feature_map = ZFeatureMap(2)

        # Convolutional building block used in QCNN
        def conv_circuit(params):
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

        # Assemble three convolution layers
        ansatz = QuantumCircuit(2)
        for i in range(3):
            layer_params = self.weight_params[4 * i : 4 * (i + 1)]
            ansatz.append(conv_circuit(layer_params), [0, 1])
            ansatz.barrier()

        # Full circuit: feature map followed by ansatz
        full_circuit = QuantumCircuit(2)
        full_circuit.append(feature_map, [0, 1])
        full_circuit.append(ansatz, [0, 1])
        return full_circuit

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return the sampler output probabilities for the given inputs."""
        return self.qnn.predict(inputs)

__all__ = ["HybridSamplerQNN"]
