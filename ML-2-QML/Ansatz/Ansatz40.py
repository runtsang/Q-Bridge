"""
RealAmplitudesAlternatingExtended

A Qiskit-compatible ansatz that extends the original alternating‑rotation
RealAmplitudes variant by adding optional diagonal (RZ) layers and a
parameter‑shared mode for these diagonals. It also allows the user to
specify an entanglement schedule per layer and to control how many
entanglement cycles are applied before each rotation sub‑layer.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the entanglement pattern.  Supported strings are
        ``"full"``, ``"linear"``, and ``"circular"``.  Alternatively a
        sequence of pairs or a callable returning such a sequence can be
        provided.

    Returns
    -------
    List[Tuple[int, int]]
        A validated list of distinct qubit pairs.

    Raises
    ------
    ValueError
        If the specification is invalid or contains out‑of‑range indices.
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


def real_amplitudes_alternating_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    diagonal_parameter_prefix: str = "phi",
    include_diagonal: bool = False,
    shared_rz: bool = False,
    entanglement_depth: int = 1,
    entanglement_schedule: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a RealAmplitudes‑type ansatz with alternating RY/RX rotations,
    optional diagonal (RZ) layers, and a per‑layer entanglement schedule.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of alternating rotation layers.  If ``skip_final_rotation_layer``
        is ``True``, the final rotation layer is omitted.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement pattern applied after each rotation layer.
    skip_final_rotation_layer : bool, default False
        Whether to omit the last rotation layer.
    insert_barriers : bool, default False
        Whether to insert barriers between rotation and entanglement blocks.
    parameter_prefix : str, default "theta"
        Prefix for the rotation parameters.
    diagonal_parameter_prefix : str, default "phi"
        Prefix for the optional diagonal (RZ) parameters.
    include_diagonal : bool, default False
        If ``True``, a diagonal RZ layer is added before each rotation layer.
    shared_rz : bool, default False
        If ``True`` and ``include_diagonal`` is set, a single RZ parameter
        is shared across all qubits in a given diagonal layer.
    entanglement_depth : int, default 1
        Number of times the entanglement pattern is applied per layer.
    entanglement_schedule : Optional[Sequence[Sequence[Tuple[int, int]]]], default None
        Optional per‑layer entanglement pairs.  Length must equal ``reps``.
    name : str | None, default None
        Custom name for the circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Raises
    ------
    ValueError
        If input arguments are invalid.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")
    if entanglement_depth < 1:
        raise ValueError("entanglement_depth must be >= 1.")
    if include_diagonal and shared_rz and not include_diagonal:
        # This condition is unreachable but kept for clarity.
        raise ValueError("shared_rz requires include_diagonal to be True.")
    if entanglement_schedule is not None:
        if len(entanglement_schedule)!= reps:
            raise ValueError(
                f"entanglement_schedule length {len(entanglement_schedule)} does not match reps {reps}."
            )
        # Validate each pair list
        for pairs in entanglement_schedule:
            for (i, j) in pairs:
                if i == j:
                    raise ValueError("Entanglement pairs must connect distinct qubits.")
                if not (0 <= i < num_qubits and 0 <= j < num_qubits):
                    raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingExtended")

    # Compute number of rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter vectors
    rotation_params = ParameterVector(parameter_prefix, num_rot_layers * n)
    diag_params: Optional[ParameterVector] = None
    if include_diagonal:
        diag_len = num_rot_layers if shared_rz else num_rot_layers * n
        diag_params = ParameterVector(diagonal_parameter_prefix, diag_len)

    def _apply_diagonal(layer: int) -> None:
        """Apply a diagonal RZ layer before the rotation sub‑layer."""
        if not include_diagonal:
            return
        if shared_rz:  # single parameter per layer
            for q in range(n):
                qc.rz(diag_params[layer], q)
        else:  # independent parameters per qubit
            base = layer * n
            for q in range(n):
                qc.rz(diag_params[base + q], q)

    def _apply_rotation(layer: int) -> None:
        """Apply the alternating RY/RX rotation sub‑layer."""
        base = layer * n
        if layer % 2 == 0:  # even layers use RY
            for q in range(n):
                qc.ry(rotation_params[base + q], q)
        else:  # odd layers use RX
            for q in range(n):
                qc.rx(rotation_params[base + q], q)

    # Main construction loop
    for r in range(reps):
        _apply_diagonal(r)
        _apply_rotation(r)
        if insert_barriers:
            qc.barrier()
        # Entanglement
        pairs = (
            entanglement_schedule[r]
            if entanglement_schedule is not None
            else _resolve_entanglement(n, entanglement)
        )
        for _ in range(entanglement_depth):
            for (i, j) in pairs:
                qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    # Final rotation layer, if not skipped
    if not skip_final_rotation_layer:
        _apply_diagonal(reps)
        _apply_rotation(reps)

    # Attach metadata
    qc.input_params = list(rotation_params) + (list(diag_params) if diag_params else [])
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingExtended(QuantumCircuit):
    """Convenience wrapper for the extended alternating‑rotation ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        diagonal_parameter_prefix: str = "phi",
        include_diagonal: bool = False,
        shared_rz: bool = False,
        entanglement_depth: int = 1,
        entanglement_schedule: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
        name: str = "RealAmplitudesAlternatingExtended",
    ) -> None:
        built = real_amplitudes_alternating_extended(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            diagonal_parameter_prefix,
            include_diagonal,
            shared_rz,
            entanglement_depth,
            entanglement_schedule,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = [
    "RealAmplitudesAlternatingExtended",
    "real_amplitudes_alternating_extended",
]
