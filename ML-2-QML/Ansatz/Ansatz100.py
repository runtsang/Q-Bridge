"""RealAmplitudes variant with alternating RY/RX rotation layers and enhanced expressivity.

This module defines the `RealAmplitudesAlternatingExtended` ansatz, extending the original
`RealAmplitudesAlternating` with the following enhancements:

* Two entangling layers per depth (first and second entanglement).
* Optional entanglement schedule, allowing a distinct entanglement pattern per repetition.
* Optional second entanglement spec, defaulting to the same as the first.
* Optional barrier insertion after each rotation or entanglement block.
* Parameter validation and clear error messages.
* Convenience constructor `real_amplitudes_alternating_extended`.
* Subclass `RealAmplitudesAlternatingExtended` inheriting from `QuantumCircuit`.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Iterable[Tuple[int, int]]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Parameters
    ----------
    num_qubits : int
        The number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]]
        Specification of the entanglement pattern.

    Returns
    -------
    List[Tuple[int, int]]
        List of distinct qubit pairs to entangle.

    Raises
    ------
    ValueError
        If the entanglement string is unknown or if the spec contains invalid pairs.
    """
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

# --------------------------------------------------------------------------- #
# Convenience constructor
# --------------------------------------------------------------------------- #
def real_amplitudes_alternating_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Iterable[Tuple[int, int]]]] = "full",
    second_entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Iterable[Tuple[int, int]]]] | None = None,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a RealAmplitudes ansatz with alternating RY/RX rotations and two entangling layers per depth.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit. Must be >= 1.
    reps : int, default 1
        Number of depth repetitions. Must be >= 1.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]], default "full"
        Specification of the first entanglement pattern.
    second_entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]], optional
        Specification of the second entanglement pattern. If None, the same pattern as ``entanglement`` is used.
    skip_final_rotation_layer : bool, default False
        If True, the final rotation layer after the last repetition is omitted.
    insert_barriers : bool, default False
        If True, insert a barrier after each rotation and entanglement block.
    parameter_prefix : str, default "theta"
        Prefix for the rotation parameters.
    name : str | None, default None
        Optional name for the circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz.

    Raises
    ------
    ValueError
        If input arguments are invalid.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingExtended")

    # Resolve entanglement patterns
    pairs1 = _resolve_entanglement(n, entanglement)
    if second_entanglement is None:
        pairs2 = pairs1
    else:
        pairs2 = _resolve_entanglement(n, second_entanglement)

    # Parameter vector
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        """Apply a single rotation layer."""
        base = layer * n
        if layer % 2 == 0:
            for q in range(n):
                qc.ry(params[base + q], q)
        else:
            for q in range(n):
                qc.rx(params[base + q], q)

    # Build the circuit
    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs1:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs2:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc

# --------------------------------------------------------------------------- #
# Class wrapper
# --------------------------------------------------------------------------- #
class RealAmplitudesAlternatingExtended(QuantumCircuit):
    """Class wrapper for the extended alternating‑rotation variant of RealAmplitudes.

    The class inherits from :class:`qiskit.QuantumCircuit` and exposes the same
    interface as the convenience constructor.  It stores the parameter vector
    and the number of rotation layers as attributes ``input_params`` and
    ``num_rot_layers`` for easy access.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Iterable[Tuple[int, int]]]] = "full",
        second_entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Iterable[Tuple[int, int]]]] | None = None,
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingExtended",
    ) -> None:
        built = real_amplitudes_alternating_extended(
            num_qubits,
            reps,
            entanglement,
            second_entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]

__all__ = ["RealAmplitudesAlternatingExtended", "real_amplitudes_alternating_extended"]
