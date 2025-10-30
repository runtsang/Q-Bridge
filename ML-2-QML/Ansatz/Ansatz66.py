"""RealAmplitudes variant with alternating RY/RX rotation layers and optional adaptive depth.

This module extends the original alternating‑rotation ansatz
by adding optional mid‑layer entanglement blocks, a configurable
adaptive layer, and a `max_depth` knob that caps the total
number of rotation layers.  The public API mirrors the seed
module: a convenience function and a subclass of `QuantumCircuit`
sharing the same name.

Typical usage

>>> qc = RealAmplitudesAlternatingExtended(4, reps=2, adaptive_layer=True)
>>> qc.draw()
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# --------------------------------------------------------------------------- #
# Helper: Resolve entanglement specification into a list of pairs
# --------------------------------------------------------------------------- #
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Translate the ``entanglement`` argument into a concrete list of two‑qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]]
        ``"full"``, ``"linear"``, ``"circular"`` or a custom list / generator.

    Returns
    -------
    List[Tuple[int, int]]
        Ordered list of unique qubit pairs.

    Raises
    ------
    ValueError
        If an invalid specification is supplied or a pair references an out‑of‑range qubit.
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
    else:
        pairs = list(entanglement)

    resolved = [(int(i), int(j)) for i, j in pairs]
    for i, j in resolved:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return resolved

# --------------------------------------------------------------------------- #
# Main ansatz builder
# --------------------------------------------------------------------------- #
def real_amplitudes_alternating_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    mid_entanglement: bool = False,
    adaptive_layer: bool = False,
    insert_barriers: bool = False,
    max_depth: int | None = None,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an alternating RY/RX rotation ansatz with optional mid‑layer entanglement
    and an adaptive final layer.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of rotation–entanglement repeats.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]], default "full"
        Specification of which qubit pairs to entangle.
    skip_final_rotation_layer : bool, default False
        Skip the extra rotation layer that would normally follow the
        ``reps`` iterations.
    mid_entanglement : bool, default False
        If ``True`` an additional entanglement block is inserted after each
        rotation layer, effectively doubling the entanglement density.
    adaptive_layer : bool, default False
        Append a final rotation–entanglement block that can be used for
        adaptive depth tuning.
    insert_barriers : bool, default False
        Insert a barrier after each logical block for visual clarity.
    max_depth : int | None, default None
        Maximum number of rotation layers (including the adaptive layer if
        ``adaptive_layer=True``).  If ``None`` no upper bound is imposed.
    parameter_prefix : str, default "theta"
        Prefix for the automatically generated parameters.
    name : str | None, default None
        Optional name for the circuit; defaults to ``"RealAmplitudesAlternatingExtended"``.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz.

    Raises
    ------
    ValueError
        If ``num_qubits`` or ``reps`` are negative, or if ``max_depth`` is
        insufficient to accommodate the requested structure.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")
    if max_depth is not None and max_depth <= 0:
        raise ValueError("max_depth must be a positive integer.")

    # Determine minimal required layers
    min_layers = (0 if skip_final_rotation_layer else 1) + (1 if adaptive_layer else 0)
    if max_depth is not None and max_depth < min_layers:
        raise ValueError(
            f"max_depth ({max_depth}) is too small for the requested structure "
            f"(requires at least {min_layers} layers)."
        )
    # Adjust reps if necessary
    if max_depth is not None:
        max_reps = max_depth - min_layers
        reps = min(reps, max_reps)

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingExtended")

    # Total number of rotation layers
    total_rot_layers = reps + (0 if skip_final_rotation_layer else 1) + (1 if adaptive_layer else 0)
    params = ParameterVector(parameter_prefix, total_rot_layers * n)

    def _rot(layer: int) -> None:
        """Apply RY or RX rotations for a given layer index."""
        base = layer * n
        if layer % 2 == 0:
            for q in range(n):
                qc.ry(params[base + q], q)
        else:
            for q in range(n):
                qc.rx(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    # Main repetition block
    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for i, j in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()
        if mid_entanglement:
            for i, j in pairs:
                qc.cx(i, j)
            if insert_barriers:
                qc.barrier()

    # Optional final rotation layer (if not skipped)
    final_layer_index = reps
    if not skip_final_rotation_layer:
        _rot(final_layer_index)
        if insert_barriers:
            qc.barrier()
        # No entanglement after the final rotation layer in the original design.

    # Optional adaptive layer
    adaptive_layer_index = reps + (0 if skip_final_rotation_layer else 1)
    if adaptive_layer:
        _rot(adaptive_layer_index)
        if insert_barriers:
            qc.barrier()
        for i, j in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()
        if mid_entanglement:
            for i, j in pairs:
                qc.cx(i, j)
            if insert_barriers:
                qc.barrier()

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = total_rot_layers  # type: ignore[attr-defined]
    return qc


# --------------------------------------------------------------------------- #
# Class wrapper
# --------------------------------------------------------------------------- #
class RealAmplitudesAlternatingExtended(QuantumCircuit):
    """
    Qiskit circuit class for the extended alternating‑rotation ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of rotation–entanglement repeats.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]], default "full"
        Entanglement schedule.
    skip_final_rotation_layer : bool, default False
        Skip the extra rotation layer after the repetitions.
    mid_entanglement : bool, default False
        Insert an additional entanglement block after each rotation.
    adaptive_layer : bool, default False
        Append a final adaptive rotation–entanglement block.
    insert_barriers : bool, default False
        Insert barriers between logical blocks.
    max_depth : int | None, default None
        Maximum number of rotation layers.
    parameter_prefix : str, default "theta"
        Prefix for parameter names.
    name : str, default "RealAmplitudesAlternatingExtended"
        Circuit name.

    Notes
    -----
    The class simply composes the circuit built by :func:`real_amplitudes_alternating_extended`
    and exposes the automatically generated parameters via ``input_params``.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        mid_entanglement: bool = False,
        adaptive_layer: bool = False,
        insert_barriers: bool = False,
        max_depth: int | None = None,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingExtended",
    ) -> None:
        built = real_amplitudes_alternating_extended(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            mid_entanglement,
            adaptive_layer,
            insert_barriers,
            max_depth,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = [
    "RealAmplitudesAlternatingExtended",
    "real_amplitudes_alternating_extended",
]
