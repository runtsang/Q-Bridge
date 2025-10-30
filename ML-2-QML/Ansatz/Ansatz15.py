"""RealAmplitudes variant with alternating RY/RX layers and optional depth‑controlled entanglement."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Union

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


def real_amplitudes_alternating_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    entangle_schedule: Sequence[
        str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
    ] | None = None,
    shared_parameters: bool | int = False,
    add_clean_up_layer: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a RealAmplitudes ansatz with alternating RY/RX rotations.

    The extended version introduces:
    * **Entanglement schedule** – a per‑layer specification of entanglement pairs.
    * **Parameter sharing** – reuse the same set of rotation parameters across layers.
    * **Clean‑up layer** – an optional final rotation layer that can be used for post‑processing.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of repeated rotation/entanglement blocks.
    entanglement : str or sequence or callable, default "full"
        Default entanglement pattern applied if ``entangle_schedule`` is not provided.
    skip_final_rotation_layer : bool, default False
        If ``True`` the final rotation layer after the last entanglement is omitted.
    insert_barriers : bool, default False
        Insert barriers between logical blocks for easier circuit inspection.
    parameter_prefix : str, default "theta"
        Prefix used for the generated parameter names.
    entangle_schedule : sequence, optional
        Sequence of entanglement specifications, one per rotation layer.
    shared_parameters : bool or int, default False
        If ``True`` all rotation layers share a single ParameterVector.
        If an ``int`` > 1, that many unique ParameterVectors are created and reused cyclically.
    add_clean_up_layer : bool, default False
        Append an additional rotation layer after all entanglement operations.
    name : str, optional
        Name of the resulting QuantumCircuit.

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

    n = int(num_qubits)

    # Determine the total number of rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    if add_clean_up_layer:
        num_rot_layers += 1

    # Resolve entanglement schedule
    if entangle_schedule is not None:
        if len(entangle_schedule)!= reps:
            raise ValueError("entangle_schedule length must match the number of repetition layers.")
    # Determine parameter set strategy
    if isinstance(shared_parameters, int) and shared_parameters > 1:
        num_param_sets = shared_parameters
    elif shared_parameters is True:
        num_param_sets = 1
    else:
        num_param_sets = num_rot_layers

    # Create parameter vectors
    param_sets: List[ParameterVector] = [
        ParameterVector(f"{parameter_prefix}_{i}", n) for i in range(num_param_sets)
    ]

    # Helper to map layer to parameter set index
    if isinstance(shared_parameters, int) and shared_parameters > 1:
        layer_to_param_index = lambda layer: layer % shared_parameters
    elif shared_parameters is True:
        layer_to_param_index = lambda layer: 0
    else:
        layer_to_param_index = lambda layer: layer

    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingExtended")

    def _rot(layer: int, params: ParameterVector) -> None:
        """Apply the alternating rotation for the given layer."""
        for q in range(n):
            if layer % 2 == 0:
                qc.ry(params[q], q)
            else:
                qc.rx(params[q], q)

    # Main loop over rotation layers
    for layer in range(reps):
        param_set_index = layer_to_param_index(layer)
        _rot(layer, param_sets[param_set_index])
        if insert_barriers:
            qc.barrier()
        ent_spec = entangle_schedule[layer] if entangle_schedule else entanglement
        pairs = _resolve_entanglement(n, ent_spec)
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    # Final rotation layer (unless skipped)
    if not skip_final_rotation_layer:
        layer = reps
        param_set_index = layer_to_param_index(layer)
        _rot(layer, param_sets[param_set_index])

    # Optional clean‑up rotation layer
    if add_clean_up_layer:
        layer = reps + (0 if skip_final_rotation_layer else 1)
        param_set_index = layer_to_param_index(layer)
        _rot(layer, param_sets[param_set_index])

    # Attach metadata
    qc.input_params = [p for ps in param_sets for p in ps]
    qc.num_rot_layers = num_rot_layers
    return qc


class RealAmplitudesAlternatingExtended(QuantumCircuit):
    """Class wrapper for the extended alternating‑rotation variant of RealAmplitudes."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        entangle_schedule: Sequence[
            str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        ] | None = None,
        shared_parameters: bool | int = False,
        add_clean_up_layer: bool = False,
        name: str = "RealAmplitudesAlternatingExtended",
    ) -> None:
        built = real_amplitudes_alternating_extended(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            entangle_schedule,
            shared_parameters,
            add_clean_up_layer,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesAlternatingExtended", "real_amplitudes_alternating_extended"]
