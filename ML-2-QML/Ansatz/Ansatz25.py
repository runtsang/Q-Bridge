"""
RealAmplitudesAlternatingExtended ansatz.

This module defines a new ansatz that extends the alternating RY/RX
variant of the RealAmplitudes circuit by adding:
    • Optional mid‑rotation layers after each entanglement sweep.
    • Additional entanglement schedules: ring‑to‑ring and star.
    • A depth multiplier that scales the number of repetitions.
    • Parameter‑sharing and barrier insertion options.

The implementation keeps the alternating rotation pattern and full/linear/circular
entanglement options from the original seed, while exposing new knobs that
increase expressivity without changing the fundamental structure.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of qubit pairs.

    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit.
    entanglement
        Either a string identifier, a sequence of pairs, or a callable that
        returns a sequence of pairs given ``num_qubits``.

    Supported string identifiers
    ----------------------------
    * ``"full"``          – all‑to‑all connectivity.
    * ``"linear"``        – nearest‑neighbour chain.
    * ``"circular"``      – linear plus a wrap‑around edge.
    * ``"ring-to-ring"``  – two passes: forward then reverse.
    * ``"star"``          – qubit 0 connected to all others.

    Returns
    -------
    List[Tuple[int, int]]
        Ordered list of two‑qubit pairs to apply CNOT gates.

    Raises
    ------
    ValueError
        If an unknown string is supplied or a pair references an out‑of‑range
        qubit.
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
        if entanglement == "ring-to-ring":
            # forward sweep
            forward = [(i, i + 1) for i in range(num_qubits - 1)]
            if num_qubits > 2:
                forward.append((num_qubits - 1, 0))
            # reverse sweep (excluding duplicate edges)
            reverse = [(i + 1, i) for i in reversed(range(1, num_qubits))]
            return forward + reverse
        if entanglement == "star":
            return [(0, i) for i in range(1, num_qubits)]
        raise ValueError(f"Unknown entanglement string: {entanglement!r}")

    if callable(entanglement):
        pairs = list(entanglement(num_qubits))
        return [(int(i), int(j)) for (i, j) in pairs]

    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(
                f"Entanglement pair {(i, j)} out of range for n={num_qubits}."
            )
    return pairs


def real_amplitudes_alternation_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    mid_rotation: bool = False,
    depth_multiplier: float = 1.0,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended alternating‑rotation RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits
        Number of qubits in the ansatz.
    reps
        Base number of repetition blocks before depth scaling.
    entanglement
        Entanglement topology; see :func:`_resolve_entanglement` for
        supported options.
    skip_final_rotation_layer
        Whether to omit the final rotation layer.
    insert_barriers
        Insert barrier gates between logical blocks for visual clarity.
    mid_rotation
        Insert an additional rotation layer immediately after each
        entanglement sweep.
    depth_multiplier
        Scale the number of repetitions. Values > 1 duplicate layers,
        values < 1 truncate layers (flooring to a minimum of 1).
    parameter_prefix
        Prefix for the parameter vector.
    name
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        Configured ansatz circuit.

    Raises
    ------
    ValueError
        For invalid input arguments.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if depth_multiplier <= 0:
        raise ValueError("depth_multiplier must be > 0.")
    n = int(num_qubits)

    # Effective number of repetitions after scaling
    effective_reps = max(1, int(round(reps * depth_multiplier)))

    # Determine total number of rotation layers
    # Pre-entanglement layers: one per repetition
    # Mid‑rotation layers: optional, one per repetition
    # Final rotation layer: optional
    pre_layers = effective_reps
    mid_layers = effective_reps if mid_rotation else 0
    final_layer = 0 if skip_final_rotation_layer else 1
    total_rot_layers = pre_layers + mid_layers + final_layer

    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingExtended")

    # Parameter vector
    params = ParameterVector(parameter_prefix, total_rot_layers * n)

    # Helper to apply a rotation layer
    def _rot(layer: int) -> None:
        base = layer * n
        if layer % 2 == 0:
            for q in range(n):
                qc.ry(params[base + q], q)
        else:
            for q in range(n):
                qc.rx(params[base + q], q)

    # Resolve entanglement schedule
    pairs = _resolve_entanglement(n, entanglement)

    # Build the circuit
    rot_layer_idx = 0
    for r in range(effective_reps):
        # Pre‑entanglement rotation
        _rot(rot_layer_idx)
        rot_layer_idx += 1

        if insert_barriers:
            qc.barrier()

        # Entanglement
        for (i, j) in pairs:
            qc.cx(i, j)

        if insert_barriers:
            qc.barrier()

        # Optional mid‑rotation
        if mid_rotation:
            _rot(rot_layer_idx)
            rot_layer_idx += 1
            if insert_barriers:
                qc.barrier()

    # Final rotation layer
    if not skip_final_rotation_layer:
        _rot(rot_layer_idx)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = total_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingExtended(QuantumCircuit):
    """
    Subclass of :class:`QuantumCircuit` implementing the extended ansatz.

    The constructor mirrors the signature of
    :func:`real_amplitudes_alternation_extended`.  Constructing an instance
    yields a ready‑to‑use circuit that can be composed, transpiled or
    parameter‑bound exactly like any other Qiskit circuit.

    Parameters
    ----------
    num_qubits, reps, entanglement, skip_final_rotation_layer,
    insert_barriers, mid_rotation, depth_multiplier,
    parameter_prefix, name
        See :func:`real_amplitudes_alternation_extended` for details.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: Union[
            str,
            Sequence[Tuple[int, int]],
            Callable[[int], Sequence[Tuple[int, int]]],
        ] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        mid_rotation: bool = False,
        depth_multiplier: float = 1.0,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingExtended",
    ) -> None:
        built = real_amplitudes_alternation_extended(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            mid_rotation,
            depth_multiplier,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = [
    "RealAmplitudesAlternatingExtended",
    "real_amplitudes_alternation_extended",
]
