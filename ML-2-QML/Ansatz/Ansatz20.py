"""Controlled modification of the RealAmplitudes ansatz with symmetry constraints and layer reordering."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

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


def real_amplitudes_controlled(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    symmetry: bool = True,
    reorder_layers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a symmetry‑constrained RealAmplitudes style circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of rotation/entanglement blocks.
    entanglement : str or sequence or callable, default "full"
        Entanglement pattern between qubits.
    skip_final_rotation_layer : bool, default False
        If True, omit the last rotation layer.
    insert_barriers : bool, default False
        Insert barriers between layers for readability.
    parameter_prefix : str, default "theta"
        Prefix for the generated parameters.
    symmetry : bool, default True
        If True, enforce mirror‑symmetry on rotation parameters, halving the parameter count.
    reorder_layers : bool, default False
        If True, alternate the order of rotation and entanglement layers to break symmetry.
    name : str, optional
        Name of the circuit.

    Returns
    -------
    QuantumCircuit
        Configured circuit with attributes ``input_params`` and ``num_rot_layers``.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")
    if not isinstance(symmetry, bool):
        raise TypeError("symmetry must be a bool.")
    if not isinstance(reorder_layers, bool):
        raise TypeError("reorder_layers must be a bool.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesControlled")

    # Determine number of unique parameters per rotation layer
    half = n // 2
    unique_params_per_layer = half + (n % 2)
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    total_params = num_rot_layers * unique_params_per_layer
    params = ParameterVector(parameter_prefix, total_params)

    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * unique_params_per_layer
        for q in range(n):
            # Mirror‑symmetric mapping
            idx = base + (q if q < half else n - 1 - q)
            qc.ry(params[idx], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        if reorder_layers and (r % 2 == 1):
            # Odd layers: entanglement first
            for (i, j) in pairs:
                qc.cx(i, j)
            if insert_barriers:
                qc.barrier()
            _rotation_layer(r)
        else:
            # Even layers or reorder_layers False: rotation first
            _rotation_layer(r)
            if insert_barriers:
                qc.barrier()
            for (i, j) in pairs:
                qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesControlled(QuantumCircuit):
    """Class‑style wrapper for the symmetry‑constrained RealAmplitudes ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        symmetry: bool = True,
        reorder_layers: bool = False,
        name: str = "RealAmplitudesControlled",
    ) -> None:
        built = real_amplitudes_controlled(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            symmetry=symmetry,
            reorder_layers=reorder_layers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesControlled", "real_amplitudes_controlled"]
