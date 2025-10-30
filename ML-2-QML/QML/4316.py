"""Quantum estimator that mirrors the classical Hybrid estimator.

The implementation builds a parameterized circuit with a graph‑style
layer structure and a quantum self‑attention block.  The circuit is
wrapped in Qiskit’s :class:`EstimatorQNN` so that it can be trained
with the same API as the classical version.  The class name
``SharedClassName`` matches the classical counterpart, enabling
easy switching between regimes.

Key design points
-----------------
* Each layer is a unitary acting on ``n`` qubits, constructed from
  parameterized RX/RY/RZ gates followed by controlled‑rotation
  entanglement.  The number of qubits equals the layer width.
* A quantum self‑attention block is inserted after each hidden layer,
  realised by a stack of controlled‑RX gates whose angles are
  trainable parameters.
* The final observable is a Pauli‑Y on the last qubit; the estimator
  returns its expectation value as the regression output.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, Iterable, Tuple

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StateEstimator


# --------------------------------------------------------------------------- #
#  Helper: build a single layer circuit
# --------------------------------------------------------------------------- #
def _build_layer(
    qr: QuantumRegister,
    layer_idx: int,
    params: Sequence[Parameter],
    use_attention: bool,
) -> QuantumCircuit:
    """
    Construct a parameterised layer with optional quantum self‑attention.

    Parameters
    ----------
    qr : QuantumRegister
        Register containing the qubits of this layer.
    layer_idx : int
        Index of the layer (used to slice parameter vector).
    params : Sequence[Parameter]
        Global list of parameters; sliced per layer.
    use_attention : bool
        Whether to include the attention block after the layer.

    Returns
    -------
    QuantumCircuit
        The constructed sub‑circuit.
    """
    num_qubits = len(qr)
    subc = QuantumCircuit(qr)

    # 1) Single‑qubit rotations
    for i in range(num_qubits):
        subc.rx(params[3 * (layer_idx * num_qubits + i)], qr[i])
        subc.ry(params[3 * (layer_idx * num_qubits + i) + 1], qr[i])
        subc.rz(params[3 * (layer_idx * num_qubits + i) + 2], qr[i])

    # 2) Entangling gates – nearest‑neighbour CRX
    for i in range(num_qubits - 1):
        subc.crx(params[3 * (layer_idx * num_qubits + num_qubits) + i], qr[i], qr[i + 1])

    # 3) Optional quantum self‑attention block
    if use_attention:
        # Simple attention: controlled‑RX with angles derived from a trainable vector
        attn_params = params[3 * (layer_idx * num_qubits + num_qubits) + (num_qubits - 1) : 3 * (layer_idx * num_qubits + num_qubits) + (num_qubits + 1)]
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                subc.crx(attn_params[i % len(attn_params)], qr[i], qr[j])

    return subc


# --------------------------------------------------------------------------- #
#  Quantum estimator class
# --------------------------------------------------------------------------- #
class SharedClassName:
    """
    Quantum analogue of the classical ``SharedClassName``.
    Builds a parameterised circuit with graph‑style layers and a
    quantum self‑attention block, then wraps it in EstimatorQNN.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer widths, e.g. ``[2, 8, 4, 1]``.
    use_attention : bool, default=True
        Whether to include the quantum self‑attention block.
    """

    def __init__(self, qnn_arch: Sequence[int], use_attention: bool = True):
        self.qnn_arch = list(qnn_arch)
        self.use_attention = use_attention

        # Compute total number of parameters
        self.param_counts = []
        total_params = 0
        for layer_idx, width in enumerate(self.qnn_arch[:-1]):
            # 3 params per qubit + (width-1) entangling params
            n_params = 3 * width + (width - 1)
            if use_attention:
                n_params += width  # one attention param per qubit
            self.param_counts.append(n_params)
            total_params += n_params

        self.param_vector = ParameterVector('theta', length=total_params)
        self.param_index = 0

        # Build quantum circuit
        self.circuit = QuantumCircuit()
        for layer_idx, width in enumerate(self.qnn_arch[:-1]):
            qr = QuantumRegister(width, f'layer{layer_idx}')
            self.circuit.add_register(qr)
            # Slice the relevant parameters
            layer_params = self.param_vector[self.param_index : self.param_index + self.param_counts[layer_idx]]
            self.param_index += self.param_counts[layer_idx]
            subc = _build_layer(qr, layer_idx, layer_params, self.use_attention)
            self.circuit.append(subc, qr)

        # Final observable – Pauli‑Y on the last qubit of the last layer
        last_qr = QuantumRegister(self.qnn_arch[-1], 'out')
        self.circuit.add_register(last_qr)
        self.circuit.h(last_qr[0])  # prepare |+> for expectation
        self.observable = qiskit.quantum_info.SparsePauliOp.from_list([( 'Y' * self.qnn_arch[-1], 1 )])

        # Backend and estimator
        backend = qiskit.Aer.get_backend('statevector_simulator')
        estimator = StateEstimator(backend)
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[],
            weight_params=self.param_vector,
            estimator=estimator,
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the quantum circuit on a batch of inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape ``(batch, input_dim)``.  The inputs are
            encoded into the circuit by applying an initial RX rotation
            to each qubit of the first layer.

        Returns
        -------
        np.ndarray
            Estimated expectation values of shape ``(batch, 1)``.
        """
        # Prepare initial rotation parameters from inputs
        init_params = np.concatenate([np.sin(inputs), np.cos(inputs)], axis=1)
        # Map to the first layer's qubits
        init_vector = init_params[:, :self.qnn_arch[0]]
        # Build parameter dictionary
        param_dict = {self.param_vector[i]: init_vector[:, i] for i in range(self.qnn_arch[0])}
        # Evaluate
        results = self.estimator_qnn.evaluate(param_dict)
        return results.reshape(-1, 1)

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit."""
        return self.circuit

__all__ = ["SharedClassName"]
