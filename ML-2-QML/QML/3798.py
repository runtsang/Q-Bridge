"""Hybrid quantum classifier that unifies the layered ansatz from the original
reference with configurable parameter clipping and an optional encoding style.

The helper keeps the same signature as the classical factory so that the
same driver code can be used for both sides:

    build_classifier_circuit(num_qubits: int, depth: int,
                             encoding_style: str = "rx",
                             clip_params: bool = True) -> Tuple[QuantumCircuit,
                                                                 Iterable,
                                                                 Iterable,
                                                                 List[SparsePauliOp]]

The returned tuple contains:

    * the parameterised circuit,
    * a list of data‑encoding parameters,
    * a list of variational parameters,
    * a list of Pauli‑Z observables for measurement.

The implementation follows the style of the original `QuantumClassifierModel`
while adding the following enhancements:

    * **Custom encodings** – the data can be encoded with RX, RY or RZ gates.
    * **Parameter clipping** – a helper is provided to clip all parameters
      before the circuit is bound to a backend.  This mirrors the clipping
      logic from the fraud‑detection example.
    * **Extensible observables** – the observable list can be extended by
      adding additional Pauli strings if needed.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to the interval ``[-bound, bound]``."""
    return max(-bound, min(bound, value))


def _clip_parameters(params: Iterable[ParameterVector], bound: float) -> None:
    """Clips all parameters in *params* in‑place."""
    for param in params:
        for i in range(len(param)):
            param[i] = _clip(float(param[i]), bound)


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    *,
    encoding_style: str = "rx",
    clip_params: bool = True,
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered variational circuit with optional encoding style and
    parameter clipping.

    Parameters
    ----------
    num_qubits:
        Number of qubits / data features.
    depth:
        Number of variational layers.
    encoding_style:
        Which single‑qubit gate to use for data encoding.  Options are
        ``"rx"``, ``"ry"``, and ``"rz"``.
    clip_params:
        If ``True`` all variational parameters are clipped to ``[-5.0, 5.0]``
        before the circuit is bound.  This mimics the clipping behaviour of
        the fraud‑detection network.

    Returns
    -------
    circuit:
        The constructed ``QuantumCircuit``.
    encoding:
        List of data‑encoding parameters.
    weights:
        List of variational parameters.
    observables:
        List of Pauli‑Z observables for each qubit.
    """
    if encoding_style not in {"rx", "ry", "rz"}:
        raise ValueError(f"Unsupported encoding_style {encoding_style!r}")

    # --------------------------------------------------------------------- #
    # Build the circuit
    # --------------------------------------------------------------------- #
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for i, param in enumerate(encoding):
        if encoding_style == "rx":
            circuit.rx(param, i)
        elif encoding_style == "ry":
            circuit.ry(param, i)
        else:  # rz
            circuit.rz(param, i)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entangling block – a simple chain of CZ gates
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Clip parameters if requested
    if clip_params:
        _clip_parameters([weights], bound=5.0)

    # Observables: one Z per qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
