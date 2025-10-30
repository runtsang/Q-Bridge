"""Extended RealAmplitudes ansatz with configurable rotation mix and entanglement scheduling."""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = ["RealAmplitudesExtended", "real_amplitudes_extended"]

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _resolve_entanglement(
    num_qubits: int,
    spec: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec."""
    if isinstance(spec, str):
        if spec == "full":
            return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        if spec == "linear":
            return [(i, i + 1) for i in range(num_qubits - 1)]
        if spec == "circular":
            pairs = [(i, i + 1) for i in range(num_qubits - 1)]
            if num_qubits > 2:
                pairs.append((num_qubits - 1, 0))
            return pairs
        raise ValueError(f"Unknown entanglement string: {spec!r}")

    if callable(spec):
        pairs = list(spec(num_qubits))
        return [(int(i), int(j)) for (i, j) in pairs]

    pairs = [(int(i), int(j)) for (i, j) in spec]  # type: ignore[arg‑type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _resolve_entanglement_schedule(
    num_qubits: int,
    spec: str
    | Sequence[Tuple[int, int]]
    | Callable[[int], Sequence[Tuple[int, int]]]
    | Sequence[Sequence[Tuple[int, int]]],
    reps: int,
) -> List[List[Tuple[int, int]]]:
    """Translate a single entanglement spec or a list of specs into a schedule of length *reps*."""
    # If the spec is a list of lists, treat each inner list as a schedule entry.
    if isinstance(spec, Iterable) and not isinstance(spec, (str, tuple, list)):
        raise TypeError("Entanglement spec must be a string, tuple, list, or callable.")
    if isinstance(spec, list) and len(spec) > 0 and isinstance(spec[0], list):
        schedule: List[List[Tuple[int, int]]] = []
        for entry in spec:
            schedule.append(_resolve_entanglement(num_qubits, entry))
    else:
        # single schedule used for all repetitions
        schedule = [_resolve_entanglement(num_qubits, spec)]

    # Pad or truncate to match the requested number of repetitions
    if len(schedule) < reps:
        schedule.extend([schedule[-1]] * (reps - len(schedule)))
    elif len(schedule) > reps:
        schedule = schedule[:reps]
    return schedule


def _apply_rotation_block(
    qc: QuantumCircuit,
    base: int,
    params: ParameterVector,
    num_qubits: int,
    include_rx: bool,
    use_u3: bool,
) -> int:
    """
    Apply a hybrid rotation block on each qubit.

    Parameters
    ----------
    qc
        The quantum circuit to append to.
    base
        Index of the first unused parameter in *params*.
    params
        Vector of parameters to use.
    num_qubits
        Number of qubits in the circuit.
    include_rx
        If True, apply an RX rotation before the RY.
    use_u3
        If True, apply a U3 rotation after the RY.

    Returns
    -------
    int
        New base index after consuming parameters.
    """
    idx = base
    for q in range(num_qubits):
        if include_rx:
            qc.rx(params[idx], q)
            idx += 1
        qc.ry(params[idx], q)
        idx += 1
        if use_u3:
            qc.u3(params[idx], params[idx + 1], params[idx + 2], q)
            idx += 3
    return idx


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def real_amplitudes_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str
    | Sequence[Tuple[int, int]]
    | Callable[[int], Sequence[Tuple[int, int]]]
    | Sequence[Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    include_rx: bool = False,
    use_u3: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended RealAmplitudes‑style ansatz.

    The ansatz interleaves rotation blocks with CX entanglers.  Each rotation
    block can optionally prepend an RX gate and/or append a U3 gate to each qubit,
    providing a richer parameterization while retaining the intuitive RY‑CX pattern.

    Parameters
    ----------
    num_qubits
        Number of qubits in the ansatz.
    reps
        Number of RY‑CX repetitions.
    entanglement
        Entanglement specification.  Accepts the same syntax as the original
        RealAmplitudes implementation, plus a list of pair lists to create a
        depth‑controlled schedule.
    skip_final_rotation_layer
        If True, omit the rotation layer following the last entanglement block.
    insert_barriers
        If True, insert a barrier before and after each entanglement block.
    include_rx
        Include an RX rotation before the RY on each qubit.
    use_u3
        Include a U3 rotation after the RY on each qubit.
    parameter_prefix
        Prefix used for automatically generated parameters.
    name
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        A parameterized ansatz circuit ready for composition or execution.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesExtended")

    # Determine rotation depth
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Compute total number of parameters
    per_qubit_params = 1 + (1 if include_rx else 0) + (3 if use_u3 else 0)
    total_params = num_rot_layers * n * per_qubit_params
    params = ParameterVector(parameter_prefix, total_params)

    # Entanglement schedule
    ent_schedule = _resolve_entanglement_schedule(n, entanglement, reps)

    # Rotation block helper
    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * n * per_qubit_params
        _ = _apply_rotation_block(
            qc,
            base,
            params,
            n,
            include_rx=include_rx,
            use_u3=use_u3,
        )

    # Build circuit
    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in ent_schedule[r]:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesExtended(QuantumCircuit):
    """Convenient subclass of :class:`qiskit.QuantumCircuit` implementing the
    extended RealAmplitudes ansatz.

    The constructor forwards all arguments to :func:`real_amplitudes_extended`
    and composes the resulting circuit in place.  Additional attributes
    ``input_params`` (the :class:`qiskit.circuit.ParameterVector` used to
    instantiate the ansatz) and ``num_rot_layers`` (the number of rotation
    layers produced) are exposed for introspection and parameter binding.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str
        | Sequence[Tuple[int, int]]
        | Callable[[int], Sequence[Tuple[int, int]]]
        | Sequence[Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        include_rx: bool = False,
        use_u3: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesExtended",
    ) -> None:
        built = real_amplitudes_extended(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            include_rx=include_rx,
            use_u3=use_u3,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
