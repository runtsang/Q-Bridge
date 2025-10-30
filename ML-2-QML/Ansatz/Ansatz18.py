"""RealAmplitudesCZExtended – a depth‑controlled, hybrid‑entanglement RealAmplitudes variant.

The design extends the original RealAmplitudes CZ‑entangling circuit by:
1. Adding an optional second rotation block per repetition (mid‑rotation).
2. Introducing a `depth` multiplier that scales the total number of repetition cycles.
3. Allowing a per‑cycle entanglement schedule via `entanglement_schedule`.
4. Supporting optional barriers for clearer visual inspection.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs based on a simple entanglement specification.

    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit (must be >= 1).
    entanglement
        Either a string keyword ('full', 'linear', 'circular'), a list of pairs,
        or a callable that returns a list of pairs given ``num_qubits``.

    Returns
    -------
    List[Tuple[int, int]]
        Valid entanglement pairs.

    Raises
    ------
    ValueError
        If an invalid string is supplied, or if any pair contains identical
        qubits or indices out of range.
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


def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    depth: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    entanglement_schedule: Sequence[
        str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
    ] | None = None,
    skip_final_rotation_layer: bool = False,
    mid_rotation: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Construct a depth‑controlled, hybrid‑entanglement RealAmplitudes CZ circuit.

    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit (must be >= 1).
    reps
        Base number of repetition cycles (must be >= 1).
    depth
        Multiplier that scales the number of repetition cycles.
    entanglement
        Default entanglement specification applied to every cycle if
        ``entanglement_schedule`` is ``None``.
    entanglement_schedule
        Optional sequence of entanglement specifications, one per cycle.
        Length must equal ``reps * depth``.
    skip_final_rotation_layer
        If ``True`` the final rotation layer after all cycles is omitted.
    mid_rotation
        If ``True`` a second rotation block is inserted immediately before each
        entanglement block, effectively doubling the number of rotation layers
        per cycle.
    insert_barriers
        If ``True`` a barrier is inserted after each rotation and entanglement
        block to aid visual inspection and debugging.
    parameter_prefix
        Prefix for the automatically generated parameters.
    name
        Optional circuit name; defaults to ``"RealAmplitudesCZExtended"``.

    Returns
    -------
    QuantumCircuit
        The constructed parameterized circuit.

    Raises
    ------
    ValueError
        If any of the inputs are invalid (e.g. negative qubit count, mismatched
        schedule length).
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if depth < 1:
        raise ValueError("depth must be >= 1.")

    n = int(num_qubits)
    total_reps = reps * depth

    # Resolve the entanglement schedule
    if entanglement_schedule is not None:
        if len(entanglement_schedule)!= total_reps:
            raise ValueError(
                f"Length of entanglement_schedule ({len(entanglement_schedule)}) "
                f"must match reps*depth ({total_reps})."
            )
    else:
        # Broadcast the default entanglement spec
        entanglement_schedule = [entanglement] * total_reps

    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    # Determine the number of rotation layers
    layers_per_cycle = 2 if mid_rotation else 1
    num_rot_layers = total_reps * layers_per_cycle
    if not skip_final_rotation_layer:
        num_rot_layers += 1

    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer_index: int) -> None:
        """Apply an RY rotation to all qubits for the specified layer."""
        base = layer_index * n
        for q in range(n):
            qc.ry(params[base + q], q)

    layer_counter = 0
    for cycle in range(total_reps):
        # Rotation(s)
        if mid_rotation:
            _rot(layer_counter)
            layer_counter += 1
            if insert_barriers:
                qc.barrier()
            _rot(layer_counter)
            layer_counter += 1
        else:
            _rot(layer_counter)
            layer_counter += 1

        if insert_barriers:
            qc.barrier()

        # Entanglement
        pairs = _resolve_entanglement(n, entanglement_schedule[cycle])
        for i, j in pairs:
            qc.cz(i, j)

        if insert_barriers:
            qc.barrier()

    # Final rotation layer
    if not skip_final_rotation_layer:
        _rot(layer_counter)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZExtended(QuantumCircuit):
    """Convenience wrapper for :func:`real_amplitudes_cz_extended`."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        depth: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        entanglement_schedule: Sequence[
            str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        ] | None = None,
        skip_final_rotation_layer: bool = False,
        mid_rotation: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            depth,
            entanglement,
            entanglement_schedule,
            skip_final_rotation_layer,
            mid_rotation,
            insert_barriers,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]
