from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
from typing import Dict, List

class HybridLayer:
    """
    Quantum counterpart of HybridLayer.  Each classical block is replaced
    with a parameterised circuit.  The run() method executes the four
    sub‑circuits sequentially and returns the measurement probabilities
    as a dictionary of NumPy arrays.
    """

    def __init__(self, conv_kernel: int = 2, n_features: int = 1, classifier_depth: int = 2):
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = 100

        # Convolutional filter (quantum analogue of Conv)
        self.conv_circuit = self._build_conv_circuit(conv_kernel)

        # Fully‑connected layer (quantum analogue of FCL)
        self.fc_circuit = self._build_fc_circuit(n_features)

        # Classifier circuit
        (
            self.classifier_circuit,
            self.classifier_encoding,
            self.classifier_weights,
            self.classifier_obs,
        ) = self._build_classifier_circuit(n_features, classifier_depth)

        # Sampler QNN (quantum analogue of SamplerQNN)
        self.sampler_qnn = self._build_sampler_qnn()

    # ----- sub‑circuit builders -------------------------------------------

    def _build_conv_circuit(self, kernel: int) -> QuantumCircuit:
        n_qubits = kernel * kernel
        qc = QuantumCircuit(n_qubits)
        params = ParameterVector("theta", n_qubits)
        for i in range(n_qubits):
            qc.rx(params[i], i)
        qc.barrier()
        qc += qiskit.circuit.random.random_circuit(n_qubits, 2)
        qc.measure_all()
        return qc

    def _build_fc_circuit(self, n_features: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        theta = ParameterVector("theta", n_features)
        qc.ry(theta[0], 0)
        qc.measure_all()
        return qc

    def _build_classifier_circuit(self, num_qubits: int, depth: int):
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)
        qc = QuantumCircuit(num_qubits)
        for param, q in zip(encoding, range(num_qubits)):
            qc.rx(param, q)
        idx = 0
        for _ in range(depth):
            for q in range(num_qubits):
                qc.ry(weights[idx], q)
                idx += 1
            for q in range(num_qubits - 1):
                qc.cz(q, q + 1)
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return qc, list(encoding), list(weights), observables

    def _build_sampler_qnn(self) -> SamplerQNN:
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
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

    # ----- execution helpers ---------------------------------------------

    def _execute(self, circuit: QuantumCircuit, param_values: List[float]) -> np.ndarray:
        bind = {param: val for param, val in zip(circuit.parameters, param_values)}
        job = qiskit.execute(
            circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bind],
        )
        result = job.result().get_counts(circuit)
        probs = np.array(list(result.values())) / self.shots
        return probs

    # ----- public API -----------------------------------------------------

    def run(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Execute the hybrid quantum pipeline.

        Parameters
        ----------
        data : dict
            Keys: 'conv', 'fc', 'classifier','sampler'.
            Each value is a flat array of parameter values matching the
            corresponding circuit.

        Returns
        -------
        dict
            Outputs from each sub‑circuit as NumPy arrays of measurement
            probabilities.
        """
        outputs: Dict[str, np.ndarray] = {}

        # Convolutional filter
        outputs["conv"] = self._execute(self.conv_circuit, data["conv"])

        # Fully‑connected layer
        outputs["fc"] = self._execute(self.fc_circuit, data["fc"])

        # Classifier: return measurement probabilities
        bind = {p: v for p, v in zip(self.classifier_circuit.parameters, data["classifier"])}
        job = qiskit.execute(
            self.classifier_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bind],
        )
        result = job.result().get_counts(self.classifier_circuit)
        probs = np.array(list(result.values())) / self.shots
        outputs["classifier"] = probs

        # Sampler QNN
        outputs["sampler"] = self.sampler_qnn.predict(data["sampler"])

        return outputs

__all__ = ["HybridLayer"]
