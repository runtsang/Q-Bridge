from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

class EstimatorQNN:
    """
    Quantum estimator that uses a Variational Quantum Circuit (VQC) as a
    regression head.  The circuit encodes the input feature vector using
    a ZZFeatureMap, then applies a small variational circuit that mimics
    the behaviour of an LSTM gate.  The outputs are sampled with a
    StatevectorSampler and interpreted as a regression value.
    """
    def __init__(self,
                 num_features: int,
                 n_qubits: int = 4,
                 reps: int = 3,
                 output_dim: int = 1) -> None:
        # Feature map
        feature_map = ZZFeatureMap(num_qubits=n_qubits, reps=reps, insert_default_gate=False)

        # Variational core – a simple layer of RX rotations and entangling CXs
        var_circuit = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            var_circuit.rx(Parameter(f"θ_{i}_rx"), i)
            var_circuit.rz(Parameter(f"θ_{i}_rz"), i)
        for i in range(n_qubits - 1):
            var_circuit.cx(i, i + 1)

        # Full circuit
        qc = QuantumCircuit(n_qubits)
        qc.append(feature_map, range(n_qubits))
        qc.append(var_circuit, range(n_qubits))
        qc.measure_all()

        # Sampler primitive
        sampler = Sampler()
        self.qnn = SamplerQNN(circuit=qc,
                              input_params=[],
                              weight_params=qc.parameters,
                              interpret=lambda x: x.mean(axis=0),
                              output_shape=(output_dim,),
                              sampler=sampler)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run the QNN on a batch of input vectors and return the sampled
        expectation values as a numpy array.
        """
        samples = self.qnn.sample(x)
        return samples.squeeze()

__all__ = ["EstimatorQNN"]
