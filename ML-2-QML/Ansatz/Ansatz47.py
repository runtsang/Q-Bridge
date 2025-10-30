"""Extended RealAmplitudes ansatz with additional rotation types and flexible entanglement.

This module builds upon the canonical RealAmplitudes ansatz by:
- Allowing arbitrary single‑qubit rotation gates per layer (RY, RZ, RX or a user‑supplied callable).
- Supporting multiple two‑qubit entanglement gates (CX, CZ, CRZ, CNOT, etc.).
- Enabling parameter sharing across qubits within a single rotation layer.
- Providing an optional per‑layer entanglement schedule.
- Inserting barriers for easier debugging and visualisation.
- Exposing both a convenience constructor `extended_real_amplitudes` and a `QuantumCircuit` subclass `ExtendedRealAmplitudes`.

The design is fully type‑checked, validates inputs with informative error messages,
and retains the same public interface as the original ansatz for seamless integration.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the entanglement pattern.  Recognised strings are
        ``"full"``, ``"linear"``, ``"circular"`` or a custom callable.
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


def _apply_rotation(
    qc: QuantumCircuit,
    gate: str | Callable[[QuantumCircuit, float, int], None],
    params: Sequence[Union[ParameterVector, ParameterVector]],
    layer_idx: int,
    n: int,
    parameter_sharing: bool,
) -> None:
    """Apply a single‑qubit rotation layer to the circuit."""
    if parameter_sharing:
        # Use the same parameter for all qubits in this layer
        for q in range(n):
            if gate == "ry":
                qc.ry(params[layer_idx], q)
            elif gate == "rz":
                qc.rz(params[layer_idx], q)
            elif gate == "rx":
                qc.rx(params[layer_idx], q)
            else:  # custom callable
                gate(qc, params[layer_idx], q)
    else:
        base = layer_idx * n
        for q in range(n):
            if gate == "ry":
                qc.ry(params[base + q], q)
            elif gate == "rz":
                qc.rz(params[base + q], q)
            elif gate == "rx":
                qc.rx(params[base + q], q)
            else:  # custom callable
                gate(qc, params[base + q], q)


def _entangle(
    qc: QuantumCircuit,
    gate: str,
    pairs: Sequence[Tuple[int, int]],
) -> None:
    """Apply a 2‑qubit entangling gate to the circuit."""
    for (i, j) in pairs:
        if gate == "cx":
            qc.cx(i, j)
        elif gate == "cz":
            qc.cz(i, j)
        elif gate == "cnot":
            qc.cnot(i, j)
        elif gate == "crz":
            # CRZ with a dummy parameter (no param needed for standard CX)
            qc.crz(0.0, i, j)
        else:
            raise ValueError(f"Unsupported entanglement gate: {gate!r}")


def extended_real_amplitudes(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    rotation_gate: str | Callable[[QuantumCircuit, float, int], None] = "ry",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    entanglement_gate: str = "cx",
    parameter_sharing: bool = False,
    entanglement_schedule: Sequence[Sequence[Tuple[int, int]]] | None = None,
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of rotation + entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement pattern.  If a sequence of pairs is supplied it is used
        for all repetitions; otherwise the spec is interpreted for each layer.
    rotation_gate : str | Callable[[QuantumCircuit, float, int], None], default "ry"
        Single‑qubit rotation gate to use in each layer.  Supported strings are
        ``"ry"``, ``"rz"``, ``"rx"`` or a custom callable that applies a rotation.
    skip_final_rotation_layer : bool, default False
        If ``True`` the final rotation layer after the last entanglement is omitted.
    insert_barriers : bool, default False
        Insert a barrier after each rotation layer and after each entanglement layer.
    parameter_prefix : str, default "theta"
        Prefix for the rotation parameters.
    entanglement_gate : str, default "cx"
        Two‑qubit entangling gate.  Supported values are ``"cx"``, ``"cz"``, ``"cnot"``, ``"crz"``.
    parameter_sharing : bool, default False
        If ``True`` the same parameter is used for all qubits within a single rotation layer.
    entanglement_schedule : Sequence[Sequence[Tuple[int, int]]] | None, default None
        Optional per‑layer entanglement patterns.  Length must equal ``reps``.
    name : str | None, default None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if rotation_gate not in {"ry", "rz", "rx"} and not callable(rotation_gate):
        raise ValueError(f"Unsupported rotation_gate: {rotation_gate!r}")

    if entanglement_gate not in {"cx", "cz", "cnot", "crz"}:
        raise ValueError(f"Unsupported entanglement_gate: {entanglement_gate!r}")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "ExtendedRealAmplitudes")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    if parameter_sharing:
        rot_params = ParameterVector(parameter_prefix, num_rot_layers)
    else:
        rot_params = ParameterVector(parameter_prefix, num_rot_layers * n)

    # Resolve entanglement pairs for each repetition
    if entanglement_schedule is not None:
        if len(entanglement_schedule)!= reps:
            raise ValueError(
                f"entanglement_schedule length {len(entanglement_schedule)} does not match reps {reps}."
            )
        ent_pairs_per_layer = [list(map(tuple, _resolve_entanglement(n, schedule))) for schedule in entanglement_schedule]
    else:
        base_pairs = _resolve_entanglement(n, entanglement)
        ent_pairs_per_layer = [base_pairs for _ in range(reps)]

    def _rotation_layer(layer_idx: int) -> None:
        _apply_rotation(qc, rotation_gate, rot_params, layer_idx, n, parameter_sharing)

    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        _entangle(qc, entanglement_gate, ent_pairs_per_layer[r])
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    qc.input_params = rot_params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class ExtendedRealAmplitudes(QuantumCircuit):
    """Convenience subclass of :class:`qiskit.circuit.QuantumCircuit` that builds
    the extended RealAmplitudes ansatz.

    The constructor forwards all arguments to :func:`extended_real_amplitudes`
    and then composes the resulting circuit onto itself.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        rotation_gate: str | Callable[[QuantumCircuit, float, int], None] = "ry",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        entanglement_gate: str = "cx",
        parameter_sharing: bool = False,
        entanglement_schedule: Sequence[Sequence[Tuple[int, int]]] | None = None,
        name: str = "ExtendedRealAmplitudes",
    ) -> None:
        built = extended_real_amplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            rotation_gate=rotation_gate,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            entanglement_gate=entanglement_gate,
            parameter_sharing=parameter_sharing,
            entanglement_schedule=entanglement_schedule,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["ExtendedRealAmplitudes", "extended_real_amplitudes"]
