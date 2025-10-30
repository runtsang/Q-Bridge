import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridFCL_QCNN:
    """
    Quantum implementation of the hybrid FCL‑QCNN architecture.
    The circuit consists of a feature map, a stack of convolutional and
    pooling layers, and a fully‑connected parameterised layer that
    acts on each qubit.
    """

    def __init__(self, n_qubits: int = 8, backend=None, shots: int = 1024) -> None:
        if backend is None:
            from qiskit import Aer
            backend = Aer.get_backend("qasm_simulator")

        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots

        estimator = Estimator()

        # ---------- Build the ansatz ----------
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

        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
            qubits = list(range(num_qubits))
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                qc.append(conv_circuit(params[param_index: param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc.append(conv_circuit(params[param_index: param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            return qc

        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def pool_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits, name="Pooling Layer")
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for source, sink in zip(range(0, num_qubits - 1, 2), range(1, num_qubits, 2)):
                qc.append(pool_circuit(params[param_index: param_index + 3]), [source, sink])
                qc.barrier()
                param_index += 3
            return qc

        # Assemble the ansatz
        ansatz = QuantumCircuit(n_qubits)
        ansatz.append(conv_layer(n_qubits, "c1"), range(n_qubits))
        ansatz.append(pool_layer(n_qubits, "p1"), range(n_qubits))
        ansatz.append(conv_layer(n_qubits // 2, "c2"), range(n_qubits // 2, n_qubits))
        ansatz.append(pool_layer(n_qubits // 2, "p2"), range(n_qubits // 2, n_qubits))
        ansatz.append(conv_layer(n_qubits // 4, "c3"), range(n_qubits // 4, n_qubits // 2))
        ansatz.append(pool_layer(n_qubits // 4, "p3"), range(n_qubits // 4, n_qubits // 2))

        # Fully‑connected layer: one RY per qubit
        fc_params = ParameterVector("fc", length=n_qubits)
        for i, p in enumerate(fc_params):
            ansatz.ry(p, i)

        # Feature map
        feature_map = ZFeatureMap(n_qubits)
        circuit = QuantumCircuit(n_qubits)
        circuit.append(feature_map, range(n_qubits))
        circuit.append(ansatz, range(n_qubits))
        circuit.measure_all()

        # Observable: sum of Z on all qubits
        observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])

        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

    def run(self, thetas: Sequence[float]) -> np.ndarray:
        """
        Evaluate the circuit with a list of parameters ``thetas`` that bind
        the fully‑connected layer.  The first ``len(feature_map.parameters)``
        elements of ``thetas`` are ignored – they are bound to zero – and
        the remaining values are mapped to the weight parameters of the
        ansatz.  The method returns the expectation value of the observable.
        """
        n_input = len(self.qnn.input_params)
        input_vals = [0.0] * n_input
        weight_vals = thetas
        if len(weight_vals)!= len(self.qnn.weight_params):
            raise ValueError(
                f"Expected {len(self.qnn.weight_params)} weight parameters, got {len(weight_vals)}."
            )
        return self.qnn(input_vals, weight_vals)

__all__ = ["HybridFCL_QCNN"]
