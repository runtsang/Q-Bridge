"""
RealAmplitudesCZExtended
========================

This module implements a deepened version of the original RealAmplitudes CZ
ansatz. The new design keeps the core rotation‑only layer and the CZ‑entangling
block, but also adds a second entangling block that can be either CZ or CNOT.
The additional entangling layer increases circuit depth and expressivity
without changing the fundamental rotation structure.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec."""
    if isinstance(entanglement, str):
        if entanglement == "full":
            return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        if entanglement == "linear":
            return [(i, i + 1) for i in range(num_qubits - 1)]
        if entanglement == "circular":
            pairs = [(i, i + 1) for i in range(num_qubits - 1)]
            if num_qubits > 2:
                pairs.append((num_qubits - 1, 0))
            return pairs
        raise ValueError(f"Unknown entanglement string: {entanglement!r}")

    if callable(entanglement):
        pairs = list(entanglement(num_qubits))
        return [(int(i), int(j)) for (i, j) in pairs]

    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _apply_entanglement(
    qc: QuantumCircuit,
    pairs: List[Tuple[int, int]],
    gate: str,
) -> None:
    """Apply a two‑qubit entangling gate to all specified pairs."""
    if gate not in {"cz", "cnot"}:
        raise ValueError(f"Unsupported entanglement gate: {gate!r}")
    for i, j in pairs:
        if gate == "cz":
            qc.cz(i, j)
        else:  # gate == "cnot"
            qc.cx(i, j)


def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    second_entanglement_type: str = "cnot",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a deepened RealAmplitudes CZ ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of rotation-entanglement cycles.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the first entanglement layer.  Accepts the same
        string options as the original seed ("full", "linear", "circular")
        or a custom list/sequence of qubit pairs.
    second_entanglement_type : str, default "cnot"
        Gate type for the second entangling block after each rotation
        layer.  Allowed values are "cz" or "cnot".
    skip_final_rotation_layer : bool, default False
        Whether to omit the final rotation layer.
    insert_barriers : bool, default False
        Whether to insert barriers between layers for readability.
    parameter_prefix : str, default "theta"
        Prefix for the rotation parameters.
    name : str | None, default None
        Optional name for the resulting QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        A Qiskit circuit implementing the extended ansatz.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")
    if second_entanglement_type not in {"cz", "cnot"}:
        raise ValueError("second_entanglement_type must be either 'cz' or 'cnot'.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        base = layer * n
        for q in range(n):
            qc.ry(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        _apply_entanglement(qc, pairs, "cz")
        if insert_barriers:
            qc.barrier()
        _apply_entanglement(qc, pairs, second_entanglement_type)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZExtended(QuantumCircuit):
    """Convenient subclass wrapping :func:`real_amplitudes_cz_extended`."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        second_entanglement_type: str = "cnot",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            entanglement,
            second_entanglement_type,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]
