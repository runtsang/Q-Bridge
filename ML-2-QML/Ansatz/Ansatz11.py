"""
RealAmplitudes‑Extended variant with alternating RY/RX layers and optional mid‑layer blocks.

This module defines a parameter‑rich circuit that expands the original
Real‑Amplitudes alternating rotation ansatz by:
- Adding a new *mid‑layer* consisting of arbitrary two‑qubit gates (default CNOT).
- Allowing the user to supply a per‑repetition entanglement schedule.
- Adding a *rotation‑only* layer that can be inserted after every mid‑entangling block.
The interface remains compatible with Qiskit, so it can directly
be composed with or bound to any other QuantumCircuit.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = [
    "RealAmplitudesAlternatingExtended",
    "real_amplitudes_alternating_extended",
]

EntanglementSpec = Union[
    str,
    Sequence[Tuple[int, int]],
    Callable[[int], Sequence[Tuple[int, int]]],
]


def _resolve_entanglement(
    num_qubits: int,
    entanglement: EntanglementSpec,
) -> List[Tuple[int, int]]:
    """Resolve a static entanglement specification into a list of qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]]
        Either a string keyword or a sequence of pairs.

    Returns
    -------
    List[Tuple[int, int]]
        A list of distinct (control, target) pairs.

    Raises
    ------
    ValueError
        If an invalid specification is provided or the pairs are out of range.
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

    # Sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(
                f"Entanglement pair {(i, j)} out of range for n={num_qubits}."
            )
    return pairs


def _rotation_layer(
    qc: QuantumCircuit,
    params: ParameterVector,
    base_idx: int,
    num_qubits: int,
    layer_idx: int,
) -> None:
    """Apply a single rotation layer (alternating RY/RX) to all qubits.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to append gates to.
    params : ParameterVector
        Parameter vector containing rotation angles.
    base_idx : int
        Index offset in the parameter vector for this layer.
    num_qubits : int
        Number of qubits.
    layer_idx : int
        Logical layer index used to decide whether to use RY or RX.
    """
    if layer_idx % 2 == 0:
        for q in range(num_qubits):
            qc.ry(params[base_idx + q], q)
    else:
        for q in range(num_qubits):
            qc.rx(params[base_idx + q], q)


def real_amplitudes_alternating_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: EntanglementSpec = "full",
    mid_entanglement: EntanglementSpec = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    insert_mid_rotation: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct an extended real‑amplitudes alternating ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    reps : int, default 1
        Number of repetition blocks.
    entanglement : EntanglementSpec, default "full"
        Entanglement pattern applied after the mid‑entanglement block.
    mid_entanglement : EntanglementSpec, default "full"
        Entanglement pattern applied before the main entanglement block.
    skip_final_rotation_layer : bool, default False
        If True, omit the rotation layer that follows the last entanglement block.
    insert_barriers : bool, default False
        If True, insert a barrier after each entanglement block.
    insert_mid_rotation : bool, default False
        If True, insert a rotation‑only layer between the mid‑entanglement and the main entanglement.
    parameter_prefix : str, default "theta"
        Prefix for parameter names.
    name : str | None, default None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        Constructed ansatz circuit.

    Raises
    ------
    ValueError
        If any input parameter is invalid.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingExtended")

    # Compute total number of rotation layers
    num_mid_rot_layers = reps if insert_mid_rotation else 0
    num_rot_layers = reps + num_mid_rot_layers + (0 if skip_final_rotation_layer else 1)
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    # Helper to resolve entanglement pairs for a given repetition
    def resolve_pairs(spec: EntanglementSpec, rep_idx: int) -> List[Tuple[int, int]]:
        if callable(spec):
            pairs = list(spec(rep_idx))
            pairs = [(int(i), int(j)) for (i, j) in pairs]
            for (i, j) in pairs:
                if i == j:
                    raise ValueError("Entanglement pairs must connect distinct qubits.")
                if not (0 <= i < n and 0 <= j < n):
                    raise ValueError(
                        f"Entanglement pair {(i, j)} out of range for n={n}."
                    )
            return pairs
        return _resolve_entanglement(n, spec)

    rot_layer_idx = 0  # counter for rotation layers

    for r in range(reps):
        # Pre‑entanglement rotation layer
        base = rot_layer_idx * n
        _rotation_layer(qc, params, base, n, rot_layer_idx)
        rot_layer_idx += 1

        # Mid‑entanglement block
        mid_pairs = resolve_pairs(mid_entanglement, r)
        for (i, j) in mid_pairs:
            qc.cx(i, j)

        # Optional mid‑rotation layer
        if insert_mid_rotation:
            base = rot_layer_idx * n
            _rotation_layer(qc, params, base, n, rot_layer_idx)
            rot_layer_idx += 1

        # Main entanglement block
        pairs = resolve_pairs(entanglement, r)
        for (i, j) in pairs:
            qc.cx(i, j)

        if insert_barriers:
            qc.barrier()

    # Optional final rotation layer
    if not skip_final_rotation_layer:
        base = rot_layer_idx * n
        _rotation_layer(qc, params, base, n, rot_layer_idx)
        rot_layer_idx += 1

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingExtended(QuantumCircuit):
    """Convenience subclass wrapping the extended ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    reps : int, default 1
        Number of repetition blocks.
    entanglement : EntanglementSpec, default "full"
        Entanglement pattern applied after the mid‑entanglement block.
    mid_entanglement : EntanglementSpec, default "full"
        Entanglement pattern applied before the main entanglement block.
    skip_final_rotation_layer : bool, default False
        If True, omit the final rotation layer.
    insert_barriers : bool, default False
        Insert barriers after each entanglement block.
    insert_mid_rotation : bool, default False
        Insert a rotation‑only layer between mid‑entanglement and main entanglement.
    parameter_prefix : str, default "theta"
        Prefix for parameter names.
    name : str, default "RealAmplitudesAlternatingExtended"
        Circuit name.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: EntanglementSpec = "full",
        mid_entanglement: EntanglementSpec = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        insert_mid_rotation: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingExtended",
    ) -> None:
        built = real_amplitudes_alternating_extended(
            num_qubits,
            reps,
            entanglement,
            mid_entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            insert_mid_rotation,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
