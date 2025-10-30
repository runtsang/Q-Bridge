"""Hybrid quantum classifier that combines a standard variational
ansatz with a quantum self‑attention subcircuit.

The module preserves the public API of the original quantum helper
`build_classifier_circuit` but expands the circuit depth to include
self‑attention gates that emulate the quantum SelfAttention block
from the reference pair.  This hybrid design offers a richer
expressivity while still being compatible with the quantum simulator
or real hardware back‑ends.

The returned tuple follows the original contract:
    (circuit, encoding_params, variational_params, observables)
where `variational_params` now also contains the parameters of the
self‑attention subcircuit.

The observables are local Z measurements on each qubit, which can be
used to construct a probability distribution over the class labels.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def _self_attention_subcircuit(qc: QuantumCircuit, n_qubits: int, start_idx: int, attention_params: np.ndarray) -> None:
    """Append a quantum self‑attention block to the existing circuit.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to which the block is appended.
    n_qubits : int
        Number of qubits in the subcircuit.
    start_idx : int
        Global index of the first parameter in the concatenated
        attention_params array.
    attention_params : np.ndarray
        Flat array of parameters: first 3*n_qubits for rotations,
        followed by (n_qubits-1) for entanglement.
    """
    rot_params = attention_params[start_idx : start_idx + 3 * n_qubits]
    ent_params = attention_params[start_idx + 3 * n_qubits : start_idx + 3 * n_qubits + (n_qubits - 1)]

    # rotation gates
    for i in range(n_qubits):
        qc.rx(rot_params[3 * i], i)
        qc.ry(rot_params[3 * i + 1], i)
        qc.rz(rot_params[3 * i + 2], i)

    # entangling gates
    for i in range(n_qubits - 1):
        qc.crx(ent_params[i], i, i + 1)


def build_classifier_circuit(num_qubits: int, depth: int, attention: bool = True) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a hybrid quantum classifier circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (i.e., feature dimensionality).
    depth : int
        Depth of the standard variational ansatz.
    attention : bool, optional
        Whether to append the self‑attention subcircuit.  When
        ``False`` the function behaves like the original reference.

    Returns
    -------
    circuit : QuantumCircuit
        The complete quantum circuit ready for simulation or execution.
    encoding_params : list[Parameter]
        Parameters used for data encoding.
    variational_params : list[Parameter]
        Concatenated parameters for the variational layers and, if
        ``attention`` is ``True``, for the attention subcircuit.
    observables : list[SparsePauliOp]
        Local Z observables on each qubit.
    """
    # data‑encoding parameters
    encoding = ParameterVector("x", num_qubits)

    # variational parameters
    var_params = ParameterVector("theta", num_qubits * depth)

    # attention parameters
    if attention:
        # 3 rotations per qubit + (n_qubits-1) entanglement gates
        attn_params = ParameterVector("alpha", 3 * num_qubits + (num_qubits - 1))
        all_params = np.concatenate([var_params, attn_params])
    else:
        all_params = var_params

    qc = QuantumCircuit(num_qubits)

    # data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    # variational ansatz
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(var_params[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # optional self‑attention subcircuit
    if attention:
        _self_attention_subcircuit(qc, num_qubits, num_qubits * depth, attn_params)

    # observables for classification
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    return qc, list(encoding), list(all_params), observables


__all__ = ["build_classifier_circuit"]
