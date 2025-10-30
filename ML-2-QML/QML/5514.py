"""ConvGen413: Quantum convolutional block.

This module merges concepts from Conv.py, EstimatorQNN.py, GraphQNN.py,
and QCNN.py.  It defines a single class ConvGen413 that builds a
parameterised quantum circuit with a feature map, convolution and pooling
layers, and evaluates it with an EstimatorQNN.  The class can be used
as a drop‑in replacement for the legacy Conv() factory in a quantum
machine‑learning pipeline.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, Tuple
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.circuit.library import ZFeatureMap, EfficientSU2
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class ConvGen413:
    """
    Quantum convolutional neural network that mirrors the classical
    ConvGen413.  The architecture follows the QCNN pattern but
    incorporates the threshold‑based data encoding from Conv.py and
    the EstimatorQNN wrapper from EstimatorQNN.py.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 100,
        threshold: int = 127,
        backend=None,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the convolution kernel (determines the number of qubits).
        shots : int
            Number of shots for the simulator.
        threshold : int
            Threshold used to encode classical data as π or 0.
        backend : qiskit.providers.Backend | None
            Backend used for execution.  Defaults to Aer qasm_simulator.
        """
        self.kernel_size = kernel_size
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Build the full quantum neural network
        self.qnn, self.feature_map, self.weight_params = self._build_ansatz()

    # ------------------------------------------------------------------
    #  Construction helpers
    # ------------------------------------------------------------------
    def _build_ansatz(self):
        """Construct a QCNN‑style ansatz with feature‑map encoding."""
        n_qubits = self.kernel_size ** 2

        # Feature‑map (amplitude encoding)
        feature_map = ZFeatureMap(n_qubits)

        # Convolution layer: 2‑qubit EfficientSU2 blocks applied pairwise
        conv_layer = QuantumCircuit(n_qubits)
        conv_params = ParameterVector("c1", length=n_qubits * 3)
        for i in range(0, n_qubits, 2):
            block = EfficientSU2(2, entanglement="linear")
            block.assign_parameters(conv_params[i : i + 3], [0, 1])
            conv_layer.append(block.to_instruction(), [i, i + 1])
        conv_layer.barrier()

        # Pooling layer: 2‑qubit EfficientSU2 blocks applied to adjacent pairs
        pool_layer = QuantumCircuit(n_qubits)
        pool_params = ParameterVector("p1", length=(n_qubits // 2) * 3)
        for i in range(0, n_qubits, 2):
            block = EfficientSU2(2, entanglement="linear")
            block.assign_parameters(pool_params[i // 2 : i // 2 + 3], [0, 1])
            pool_layer.append(block.to_instruction(), [i, i + 1])
        pool_layer.barrier()

        # Full ansatz: feature‑map → convolution → pooling
        full = QuantumCircuit(n_qubits)
        full.compose(feature_map, list(range(n_qubits)), inplace=True)
        full.compose(conv_layer, list(range(n_qubits)), inplace=True)
        full.compose(pool_layer, list(range(n_qubits)), inplace=True)

        # Observable for regression (single Z on first qubit)
        observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])

        # EstimatorQNN wrapper
        estimator = StatevectorEstimator()
        weight_params = full.parameters  # all trainable params
        input_params = feature_map.parameters

        qnn = EstimatorQNN(
            circuit=full,
            observables=observable,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
        )
        return qnn, feature_map, weight_params

    # ------------------------------------------------------------------
    #  Execution
    # ------------------------------------------------------------------
    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum neural network on the supplied data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (H, W) with values in [0, 255].

        Returns
        -------
        float
            Expectation value of the observable.
        """
        # Encode data as angles for the feature map
        param_binds = []
        for val in data.flatten():
            angle = np.pi if val > self.threshold else 0.0
            param_binds.append({p: angle for p in self.feature_map.parameters})

        # Weight parameters are fixed to zero for inference
        weight_binds = {p: 0.0 for p in self.weight_params}

        # Run the estimator
        expectation = self.qnn.run(
            inputs=param_binds,
            weight_params=weight_binds,
        )
        return expectation

    # ------------------------------------------------------------------
    #  Utility: random graph network (mirrors GraphQNN.random_network)
    # ------------------------------------------------------------------
    @staticmethod
    def random_graph_network(
        qnn_arch: Sequence[int], samples: int
    ) -> Tuple[Sequence[int], list[list[object]], list[Tuple[object, object]], object]:
        """
        Generate a random quantum network and training data.

        This method is a quantum analogue of GraphQNN.random_network
        and returns the network architecture, a list of random unitary
        layers, a dataset of input‑output state pairs, and the target
        unitary.
        """
        from qutip import Qobj, tensor, random_unitary, fock, qeye
        import scipy as sc

        num_layers = len(qnn_arch) - 1
        unitaries: list[list[Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: list[Qobj] = []
            for output in range(num_outputs):
                op = random_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = tensor(op, qeye(num_outputs - 1))
                    # Swap registers if needed (omitted for brevity)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        # Target unitary
        target_unitary = random_unitary(qnn_arch[-1])

        # Training data: random input states mapped by target
        dataset: list[Tuple[Qobj, Qobj]] = []
        for _ in range(samples):
            state = Qobj(sc.random.normal(size=(2 ** qnn_arch[-1], 1)) + 1j * sc.random.normal(size=(2 ** qnn_arch[-1], 1)))
            state /= sc.linalg.norm(state.data)
            dataset.append((state, target_unitary * state))

        return list(qnn_arch), unitaries, dataset, target_unitary

__all__ = ["ConvGen413"]
