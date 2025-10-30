"""Extended RealAmplitudes ansatz with optional additional rotations, entanglement schedules, and parameter sharing.

This module defines a function ``real_amplitudes_extended`` and a convenience subclass
``RealAmplitudesExtended`` that build a parameterised circuit similar to Qiskit's
``RealAmplitudes`` but with several extensions:
* Optional RZ rotation after each RY layer.
* Parameter sharing across repetitions.
* Custom entanglement schedules per repetition.
* Optional SWAP gates after entangling gates.
* Barrier insertion between layers.
The design keeps the original intuition of RY rotations followed by CX entanglers
while allowing the user to tune expressivity and connectivity.

"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# --------------------------------------------------------------------------- #
# Helper functions for entanglement resolution
# --------------------------------------------------------------------------- #
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported string patterns:
    * ``"full"`` – all qubit pairs
    * ``"linear"`` – nearest‑neighbour chain
    * ``"circular"`` – linear chain plus a connection between the last and first qubit
    * ``"alternating"`` – pairs (0,1), (2,3), … (last qubit unpaired if odd).

    ``entanglement`` may also be a user‑provided sequence of pairs.  Each pair
    must connect distinct qubits and lie within the valid qubit range.
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
        if entanglement == "alternating":
            return [(i, i + 1) for i in range(0, num_qubits - 1, 2)]
        raise ValueError(f"Unknown entanglement string: {entanglement!r}")

    # Assume a sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _resolve_entanglement_schedule(
    num_qubits: int,
    reps: int,
    entanglement: str | Sequence[Tuple[int, int]] | Sequence[Sequence[Tuple[int, int]]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[List[Tuple[int, int]]]:
    """Return a list of entanglement pairs for each repetition.

    The ``entanglement`` argument can be:
    * a string pattern (see :func:`_resolve_entanglement`) – applied to all repetitions.
    * a callable ``f(num_qubits)`` – invoked for each repetition.
    * a sequence of pairs – applied uniformly to all repetitions.
    * a sequence of sequences of pairs – a custom schedule per repetition.
    """
    if isinstance(entanglement, str):
        pairs = _resolve_entanglement(num_qubits, entanglement)
        return [pairs for _ in range(reps)]

    if callable(entanglement):
        return [list(entanglement(num_qubits)) for _ in range(reps)]

    if isinstance(entanglement, Sequence):
        first = entanglement[0]
        # Sequence of pairs
        if isinstance(first, tuple):
            pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
            for (i, j) in pairs:
                if i == j:
                    raise ValueError("Entanglement pairs must connect distinct qubits.")
                if not (0 <= i < num_qubits and 0 <= j < num_qubits):
                    raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
            return [pairs for _ in range(reps)]

        # Custom schedule: sequence of sequences of pairs
        if len(entanglement)!= reps:
            raise ValueError("Custom entanglement schedule length must match number of repetitions.")
        schedule: List[List[Tuple[int, int]]] = []
        for pair_list in entanglement:
            if not isinstance(pair_list, Sequence):
                raise ValueError("Each element of entanglement schedule must be a sequence of tuples.")
            pairs = [(int(i), int(j)) for (i, j) in pair_list]
            for (i, j) in pairs:
                if i == j:
                    raise ValueError("Entanglement pairs must connect distinct qubits.")
                if not (0 <= i < num_qubits and 0 <= j < num_qubits):
                    raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
            schedule.append(pairs)
        return schedule

    raise TypeError("Entanglement must be a string, a sequence of pairs, a sequence of schedules, or a callable.")


# --------------------------------------------------------------------------- #
# Extended ansatz builder
# --------------------------------------------------------------------------- #
def real_amplitudes_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Sequence[Sequence[Tuple[int, int]]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
    *,
    extra_rotation: bool = False,
    parameter_sharing: bool = False,
    swap_entanglement: bool = False,
) -> QuantumCircuit:
    """
    Build an extended RealAmplitudes-style ``QuantumCircuit``.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default: 1
        Number of RY rotation + entanglement repetitions.
    entanglement : str or sequence or callable
        Defines the two‑qubit entanglement pattern.  See
        :func:`_resolve_entanglement_schedule` for details.
    skip_final_rotation_layer : bool, default: False
        If ``True`` the final rotation layer after the last entanglement is omitted.
    insert_barriers : bool, default: False
        Insert a barrier between layers for visual clarity.
    parameter_prefix : str, default: "theta"
        Prefix used for the parameter vector names.
    name : str or None, default: None
        Optional circuit name.  If ``None`` a default is chosen.
    extra_rotation : bool, default: False
        If ``True`` an additional RZ rotation is applied after each RY rotation.
    parameter_sharing : bool, default: False
        If ``True`` all rotation layers share the same parameters.
    swap_entanglement : bool, default: False
        If ``True`` a SWAP gate is inserted immediately after each CX gate.

    Returns
    -------
    QuantumCircuit
        The constructed circuit.  The circuit exposes ``input_params`` and
        ``num_rot_layers`` attributes for introspection.

    Notes
    -----
    * The circuit preserves the canonical RY‑CX structure.
    * When ``parameter_sharing`` is ``True`` the parameter vector size is
      ``n * (1 + extra_rotation)``; otherwise it is this size multiplied by
      the number of rotation layers.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if not isinstance(parameter_prefix, str):
        raise TypeError("parameter_prefix must be a string.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesExtended")

    # Resolve entanglement schedule
    entanglement_schedule = _resolve_entanglement_schedule(n, reps, entanglement)

    # Compute parameter vector size
    base_params_per_layer = n * (1 + int(extra_rotation))
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    total_params = base_params_per_layer if parameter_sharing else base_params_per_layer * num_rot_layers
    params = ParameterVector(parameter_prefix, total_params)

    def _rotation_layer(layer_idx: int) -> None:
        """Apply the rotation layer for a given repetition."""
        base = 0 if parameter_sharing else layer_idx * base_params_per_layer
        for q in range(n):
            qc.ry(params[base + q], q)
            if extra_rotation:
                qc.rz(params[base + n + q], q)

    # Build the circuit
    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in entanglement_schedule[r]:
            qc.cx(i, j)
            if swap_entanglement:
                qc.swap(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    # Attach metadata for introspection
    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    qc.entanglement_schedule = entanglement_schedule  # type: ignore[attr-defined]
    qc.parameter_sharing = parameter_sharing  # type: ignore[attr-defined]
    qc.extra_rotation = extra_rotation  # type: ignore[attr-defined]
    qc.swap_entanglement = swap_entanglement  # type: ignore[attr-defined]
    return qc


# --------------------------------------------------------------------------- #
# Class wrapper
# --------------------------------------------------------------------------- #
class RealAmplitudesExtended(QuantumCircuit):
    """Convenience subclass that builds an extended RealAmplitudes ansatz.

    The constructor forwards all arguments to :func:`real_amplitudes_extended`
    and composes the resulting circuit into ``self``.  The instance exposes
    the same metadata attributes as the builder function.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Sequence[Sequence[Tuple[int, int]]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesExtended",
        *,
        extra_rotation: bool = False,
        parameter_sharing: bool = False,
        swap_entanglement: bool = False,
    ) -> None:
        built = real_amplitudes_extended(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            name=name,
            extra_rotation=extra_rotation,
            parameter_sharing=parameter_sharing,
            swap_entanglement=swap_entanglement,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        # Preserve metadata
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.entanglement_schedule = built.entanglement_schedule  # type: ignore[attr-defined]
        self.parameter_sharing = built.parameter_sharing  # type: ignore[attr-defined]
        self.extra_rotation = built.extra_rotation  # type: ignore[attr-defined]
        self.swap_entanglement = built.swap_entanglement  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesExtended", "real_amplitudes_extended"]
