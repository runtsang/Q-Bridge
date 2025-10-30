"""
Quantum counterpart of the hybrid QCNN model.
Builds a feature‑map‑driven variational circuit that combines
convolutional, pooling, and a single‑qubit EstimatorQNN unit.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
import torchquantum as tq


class QCNNHybridModel(tq.QuantumModule):
    """
    Quantum module that mirrors the classical hybrid architecture.
    The circuit consists of:
        - ZFeatureMap(8) as input feature map.
        - Three convolutional layers (2‑qubit blocks) with pooling.
        - A single‑qubit EstimatorQNN unit (h, ry, rx).
    """

    def __init__(self) -> None:
        super().__init__()
        self.qnn = self._build_qnn()

    def _build_qnn(self) -> EstimatorQNN:
        estimator = Estimator()

        # --- Convolution block (same as QCNN.py) ---
        def conv_circuit(params):
            target = QuantumCircuit(2)
            target.rz(-np.pi / 2, 1)
            target.cx(1, 0)
            target.rz(params[0], 0)
            target.ry(params[1], 1)
            target.cx(0, 1)
            target.ry(params[2], 1)
            target.cx(1, 0)
            target.rz(np.pi / 2, 0)
            return target

        def pool_circuit(params):
            target = QuantumCircuit(2)
            target.rz(-np.pi / 2, 1)
            target.cx(1, 0)
            target.rz(params[0], 0)
            target.ry(params[1], 1)
            target.cx(0, 1)
            target.ry(params[2], 1)
            return target

        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
            qubits = list(range(num_qubits))
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                qc = qc.compose(
                    conv_circuit(params[param_index : param_index + 3]),
                    [q1, q2],
                )
                qc.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc = qc.compose(
                    conv_circuit(params[param_index : param_index + 3]),
                    [q1, q2],
                )
                qc.barrier()
                param_index += 3
            qc_inst = qc.to_instruction()
            qc = QuantumCircuit(num_qubits)
            qc.append(qc_inst, qubits)
            return qc

        def pool_layer(sources, sinks, param_prefix):
            num_qubits = len(sources) + len(sinks)
            qc = QuantumCircuit(num_qubits, name="Pooling Layer")
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for source, sink in zip(sources, sinks):
                qc = qc.compose(
                    pool_circuit(params[param_index : param_index + 3]),
                    [source, sink],
                )
                qc.barrier()
                param_index += 3
            qc_inst = qc.to_instruction()
            qc = QuantumCircuit(num_qubits)
            qc.append(qc_inst, range(num_qubits))
            return qc

        # Feature map
        feature_map = ZFeatureMap(8)

        # Ansatzz construction
        ansatz = QuantumCircuit(8, name="Ansatz")
        ansatz.compose(conv_layer(8, "c1"), inplace=True)
        ansatz.compose(
            pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True
        )
        ansatz.compose(conv_layer(4, "c2"), inplace=True)
        ansatz.compose(
            pool_layer([0, 1], [2, 3], "p2"), inplace=True
        )
        ansatz.compose(conv_layer(2, "c3"), inplace=True)
        ansatz.compose(
            pool_layer([0], [1], "p3"), inplace=True
        )

        # EstimatorQNN‑style single‑qubit unit
        single = QuantumCircuit(1, name="EstimatorQNN Unit")
        theta_w = Parameter("theta_w")
        single.h(0)
        single.ry(theta_w, 0)
        single.rx(theta_w, 0)
        single_inst = single.to_instruction()
        ansatz.append(single_inst, [0])  # attach to qubit 0

        # Combine feature map and ansatz
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        # Build EstimatorQNN
        qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters + [theta_w],
            estimator=estimator,
        )
        return qnn

    def forward(self, inputs: dict) -> torch.Tensor:
        """
        Forward pass for the quantum hybrid model.

        Parameters
        ----------
        inputs : dict
            Mapping from qiskit Parameter objects to numeric values.
            Typically `{input_param: value}` for the feature map.

        Returns
        -------
        torch.Tensor
            Expectation value of the observable.
        """
        return self.qnn.forward(inputs)


__all__ = ["QCNNHybridModel"]
