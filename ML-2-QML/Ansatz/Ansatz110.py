"""RealAmplitudes variant with alternating RY/RX rotation layers and configurable hybrid entanglement."""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = ["RealAmplitudesAlternatingExtended", "real_amplitudes_alternating_extended"]

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve a user‑supplied entanglement specification into a list of two‑qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Either a keyword ('full', 'linear', 'circular'), a static list of pairs,
        or a callable that returns a list given the qubit count.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of distinct qubit pairs.

    Raises
    ------
    ValueError
        If an invalid specification is provided.
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
        raise ValueError(f"Unknown entanglement keyword: {entanglement!r}")

    if callable(entanglement):
        pairs = list(entanglement(num_qubits))
        return [(int(i), int(j)) for i, j in pairs]

    # Assume a static sequence of pairs
    pairs = [(int(i), int(j)) for i, j in entanglement]  # type: ignore[arg-type]
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def real_amplitudes_alternating_extended(
    num_qubits: int,
    reps: int = 1,
    depth_multiplier: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    entanglement_schedule: Sequence[Sequence[Tuple[int, int]]] | Callable[[int, int], Sequence[Tuple[int, int]]] | None = None,
    use_parametric_entanglement: bool = False,
    skip_final_rotation_layer: bool | Sequence[bool] = False,
    barrier_every: int | None = None,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a depth‑controlled alternating‑rotation ansatz with optional parametric entanglement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int
        Base number of rotation layers.
    depth_multiplier : int, default 1
        Multiplies the base `reps` to obtain the total number of repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Default entanglement pattern applied when a schedule is not supplied.
    entanglement_schedule : Sequence[Sequence[Tuple[int, int]]] | Callable[[int, int], Sequence[Tuple[int, int]]] | None
        Optional per‑repetition schedule of entanglement pairs.
    use_parametric_entanglement : bool, default False
        If True, applies a CRZ gate with a dedicated parameter for each pair.
    skip_final_rotation_layer : bool | Sequence[bool], default False
        Flag to skip the final rotation layer. Can be a single boolean applied to all repetitions
        or a list of booleans indexed by repetition.
    barrier_every : int | None, default None
        Insert a barrier after every `barrier_every` repetitions.
    insert_barriers : bool, default False
        Insert a barrier after each rotation/entanglement block.
    parameter_prefix : str, default "theta"
        Prefix for rotation parameters.
    name : str | None, default None
        Circuit name.

    Returns
    -------
    QuantumCircuit
        Configurable alternating‑rotation ansatz.

    Notes
    -----
    - The total number of repetitions is `total_reps = reps * depth_multiplier`.
    - When `use_parametric_entanglement` is True, an additional parameter per entanglement pair
      is added to the circuit.
    - The `skip_final_rotation_layer` can be a list matching `total_reps`; if shorter, it is
      padded with the last value.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    total_reps = reps * depth_multiplier
    n = int(num_qubits)

    # Determine rotation parameters
    rot_params = ParameterVector(parameter_prefix, total_reps * n)

    # Determine entanglement parameters
    ent_params: ParameterVector | None = None
    if use_parametric_entanglement:
        # Count total pairs across all repetitions
        pair_counts: List[int] = []
        for r in range(total_reps):
            pairs = _resolve_entanglement(
                n,
                entanglement_schedule[r] if entanglement_schedule and isinstance(entanglement_schedule, list) else entanglement,
            )
            pair_counts.append(len(pairs))
        total_pairs = sum(pair_counts)
        ent_params = ParameterVector(f"{parameter_prefix}_ent", total_pairs)

    # Prepare skip flags
    if isinstance(skip_final_rotation_layer, bool):
        skip_flags = [skip_final_rotation_layer] * total_reps
    else:
        skip_flags = list(skip_final_rotation_layer)
        if len(skip_flags) < total_reps:
            skip_flags.extend([skip_flags[-1]] * (total_reps - len(skip_flags)))

    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingExtended")

    ent_param_index = 0
    for r in range(total_reps):
        base = r * n
        # Rotation layer
        if r % 2 == 0:
            # Even layers: RY
            for q in range(n):
                qc.ry(rot_params[base + q], q)
        else:
            # Odd layers: RX
            for q in range(n):
                qc.rx(rot_params[base + q], q)

        if insert_barriers:
            qc.barrier()

        # Entanglement layer
        pairs = _resolve_entanglement(
            n,
            entanglement_schedule[r] if entanglement_schedule and isinstance(entanglement_schedule, list) else entanglement,
        )
        for i, j in pairs:
            if use_parametric_entanglement:
                qc.crz(ent_params[ent_param_index], i, j)
                ent_param_index += 1
            else:
                qc.cx(i, j)

        if insert_barriers:
            qc.barrier()

        # Optional final rotation layer for this repetition
        if not skip_flags[r]:
            base = r * n
            if r % 2 == 0:
                for q in range(n):
                    qc.ry(rot_params[base + q], q)
            else:
                for q in range(n):
                    qc.rx(rot_params[base + q], q)

        if barrier_every and (r + 1) % barrier_every == 0:
            qc.barrier()

    # Attach parameter metadata
    qc.input_params = rot_params  # type: ignore[attr-defined]
    qc.num_rot_layers = total_reps  # type: ignore[attr-defined]
    if use_parametric_entanglement:
        qc.ent_params = ent_params  # type: ignore[attr-defined]
        qc.num_entanglement_params = ent_param_index  # type: ignore[attr-defined]

    return qc


class RealAmplitudesAlternatingExtended(QuantumCircuit):
    """Convenience subclass exposing the extended alternating‑rotation ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        depth_multiplier: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        entanglement_schedule: Sequence[Sequence[Tuple[int, int]]] | Callable[[int, int], Sequence[Tuple[int, int]]] | None = None,
        use_parametric_entanglement: bool = False,
        skip_final_rotation_layer: bool | Sequence[bool] = False,
        barrier_every: int | None = None,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingExtended",
    ) -> None:
        built = real_amplitudes_alternating_extended(
            num_qubits,
            reps,
            depth_multiplier,
            entanglement,
            entanglement_schedule,
            use_parametric_entanglement,
            skip_final_rotation_layer,
            barrier_every,
            insert_barriers,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        if hasattr(built, "ent_params"):
            self.ent_params = built.ent_params  # type: ignore[attr-defined]
            self.num_entanglement_params = built.num_entanglement_params  # type: ignore[attr-defined]
