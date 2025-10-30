from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN as BaseSamplerQNN

class HybridAutoencoder(BaseSamplerQNN):
    """Quantum encoder for the hybrid autoencoder using a QCNN ansatz."""
    def __init__(self, latent_dim: int = 3, shots: int = 1024):
        # The number of qubits equals the latent dimension
        num_qubits = latent_dim

        def _conv_circuit(params):
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

        def _pool_circuit(params):
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
                qc = qc.compose(_conv_circuit(params[param_index:param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc = qc.compose(_conv_circuit(params[param_index:param_index + 3]), [q1, q2])
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
                qc = qc.compose(_pool_circuit(params[param_index:param_index + 3]), [source, sink])
                qc.barrier()
                param_index += 3
            qc_inst = qc.to_instruction()
            qc = QuantumCircuit(num_qubits)
            qc.append(qc_inst, range(num_qubits))
            return qc

        # Feature map
        feature_map = ZFeatureMap(num_qubits)
        # Ansätze construction
        ansatz = QuantumCircuit(num_qubits)
        ansatz.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p1"), range(num_qubits), inplace=True)
        ansatz.compose(conv_layer(num_qubits // 2, "c2"), range(num_qubits // 2, num_qubits), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), range(num_qubits // 2, num_qubits), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(num_qubits)
        circuit.compose(feature_map, range(num_qubits), inplace=True)
        circuit.compose(ansatz, range(num_qubits), inplace=True)

        sampler = Sampler()
        super().__init__(
            circuit=circuit,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            sampler=sampler,
            interpret=lambda x: x,
            output_shape=(latent_dim,),
        )
