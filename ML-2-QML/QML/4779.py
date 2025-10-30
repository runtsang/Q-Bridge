"""Quantum hybrid self‑attention model using Qiskit.

The quantum circuit is a variational ansatz that merges attention gates
and QCNN layers into a single circuit.  A Z‑feature map encodes the
classical input data and the circuit is evaluated with the Statevector
simulator.  The class exposes an ``evaluate`` method that returns the
expectation value of a single‑qubit Z observable for a given parameter
set and input data.
"""

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

__all__ = ["QuantumHybridSelfAttention"]

class QuantumHybridSelfAttention:
    """
    Quantum self‑attention circuit that merges attention gates
    and QCNN layers into a single variational ansatz.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    num_layers : int, optional
        Number of convolution‑pooling layers to stack.
    """
    def __init__(self, n_qubits: int = 4, num_layers: int = 3) -> None:
        self.n_qubits = n_qubits
        self.num_layers = num_layers

        # Feature map that encodes the classical input
        self.feature_map = ZFeatureMap(n_qubits)

        # Build the variational circuit
        self.circuit, self.weight_params = self._build_ansatz()

        # Observable: Z on the first qubit
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])

        # Estimator that will evaluate the circuit
        self.estimator = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.weight_params
        )

    def _attention_subcircuit(self, params: ParameterVector) -> QuantumCircuit:
        """
        Build an attention‑style sub‑circuit with rotations and CRX gates.
        """
        circ = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            circ.rx(params[3 * i], i)
            circ.ry(params[3 * i + 1], i)
            circ.rz(params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(params[self.n_qubits * 3 + i], i, i + 1)
        return circ

    def _conv_layer(self, params: ParameterVector) -> QuantumCircuit:
        """
        Convolutional layer inspired by the QCNN reference.
        """
        circ = QuantumCircuit(self.n_qubits)
        for i in range(0, self.n_qubits - 1, 2):
            circ.cx(i, i + 1)
            circ.rz(params[i], i)
            circ.ry(params[i + 1], i + 1)
        return circ

    def _pool_layer(self, params: ParameterVector) -> QuantumCircuit:
        """
        Pooling layer that reduces the number of qubits via a simple CX‑RZ sequence.
        """
        circ = QuantumCircuit(self.n_qubits)
        for i in range(0, self.n_qubits - 1, 2):
            circ.cx(i, i + 1)
            circ.rz(params[i], i)
        return circ

    def _build_ansatz(self) -> tuple[QuantumCircuit, list[ParameterVector]]:
        """
        Construct the full variational ansatz.

        Returns
        -------
        tuple[QuantumCircuit, list[ParameterVector]]
            The ansatz circuit and a list of weight parameters.
        """
        ansatz = QuantumCircuit(self.n_qubits)
        weight_params = []

        # Feature map
        ansatz.append(self.feature_map, range(self.n_qubits))

        for l in range(self.num_layers):
            # Attention sub‑circuit
            attn_params = ParameterVector(f"attn_{l}", length=self.n_qubits * 3 + self.n_qubits - 1)
            weight_params.append(attn_params)
            ansatz.append(self._attention_subcircuit(attn_params), range(self.n_qubits))

            # Convolutional layer
            conv_params = ParameterVector(f"conv_{l}", length=self.n_qubits)
            weight_params.append(conv_params)
            ansatz.append(self._conv_layer(conv_params), range(self.n_qubits))

            # Pooling layer
            pool_params = ParameterVector(f"pool_{l}", length=self.n_qubits)
            weight_params.append(pool_params)
            ansatz.append(self._pool_layer(pool_params), range(self.n_qubits))

        return ansatz, weight_params

    def evaluate(self,
                 rotation_params: np.ndarray,
                 entangle_params: np.ndarray,
                 theta: np.ndarray,
                 input_data: np.ndarray) -> np.ndarray:
        """
        Evaluate the expectation value of the observable for a single input sample.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation parameters for the attention sub‑circuit.
        entangle_params : np.ndarray
            Entanglement parameters for the attention sub‑circuit.
        theta : np.ndarray
            Ansatz parameters (concatenated across layers).
        input_data : np.ndarray
            Classical input data to be encoded by the feature map.

        Returns
        -------
        np.ndarray
            Expectation value of the observable.
        """
        # Build parameter dictionary
        param_dict = {}
        idx = 0
        for l in range(self.num_layers):
            attn_len = self.n_qubits * 3 + self.n_qubits - 1
            for i in range(attn_len):
                param_dict[f"attn_{l}_{i}"] = theta[idx]
                idx += 1
            conv_len = self.n_qubits
            for i in range(conv_len):
                param_dict[f"conv_{l}_{i}"] = theta[idx]
                idx += 1
            pool_len = self.n_qubits
            for i in range(pool_len):
                param_dict[f"pool_{l}_{i}"] = theta[idx]
                idx += 1

        # Feature map parameters
        for p, val in zip(self.feature_map.parameters, input_data):
            param_dict[str(p)] = val

        return self.estimator.evaluate(param_dict).reshape(-1)
