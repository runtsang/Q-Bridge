"""RealAmplitudes variant with alternating RY/RX rotation layers, optional mid‑layer rotations, and tunable entanglement depth."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter

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
    entanglement_depth: int = 1,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    insert_mid_rotations: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a RealAmplitudes ansatz with alternating RY/RX rotation layers,
    optional mid‑layer rotations, and a tunable entanglement depth.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of repetition blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Specification of the two‑qubit entanglement pattern.
    entanglement_depth : int, default 1
        Number of consecutive entanglement layers applied within each repetition.
    skip_final_rotation_layer : bool, default False
        If True, omit the final rotation layer after the last repetition.
    insert_barriers : bool, default False
        If True, insert barriers between logical blocks for readability.
    insert_mid_rotations : bool, default False
        If True, insert an additional rotation layer between the main rotation
        layer and the entanglement layer within each repetition.
    parameter_prefix : str, default "theta"
        Prefix for the rotation parameters.
    name : str | None, default None
        Name of the circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be >= 0.")
    if entanglement_depth < 1:
        raise ValueError("entanglement_depth must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingExtended")

    # Compute total number of rotation layers
    num_rot_layers = reps
    if insert_mid_rotations:
        num_rot_layers += reps
    if not skip_final_rotation_layer:
        num_rot_layers += 1

    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        base = layer * n
        if layer % 2 == 0:
            for q in range(n):
                qc.ry(params[base + q], q)
        else:
            for q in range(n):
                qc.rx(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    layer_idx = 0
    for r in range(reps):
        _rot(layer_idx)          # main rotation layer
        layer_idx += 1
        if insert_mid_rotations:
            _rot(layer_idx)      # mid rotation layer
            layer_idx += 1
        # entanglement depth
        for _ in range(entanglement_depth):
            for (i, j) in pairs:
                qc.cx(i, j)
            if insert_barriers:
                qc.barrier()
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(layer_idx)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc

class RealAmplitudesAlternatingExtended(QuantumCircuit):
    """Class wrapper for the extended alternating‑rotation ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        entanglement_depth: int = 1,
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        insert_mid_rotations: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingExtended",
    ) -> None:
        built = real_amplitudes_alternating_extended(
            num_qubits,
            reps,
            entanglement,
            entanglement_depth,
            skip_final_rotation_layer,
            insert_barriers,
            insert_mid_rotations,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]

__all__ = ["RealAmplitudesAlternatingExtended", "real_amplitudes_alternating_extended"]
