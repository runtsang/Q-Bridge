"""RealAmplitudesSymmetry ansatz with symmetry constraints and optional parameter sharing."""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

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


def real_amplitudes_symmetric(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    share_params_across_layers: bool = False,
    symmetry: bool = True,
    name: str | None = None,
) -> QuantumCircuit:
    """Build a symmetry‑constrained RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits
        Number of qubits in the ansatz.
    reps
        Number of entanglement blocks.
    entanglement
        Entanglement topology or callable returning pairs.
    skip_final_rotation_layer
        If True, omit the final rotation layer.
    insert_barriers
        Insert barriers after each rotation/entanglement block for readability.
    parameter_prefix
        Prefix for parameter names.
    share_params_across_layers
        If True, all rotation layers share the same parameter set.
    symmetry
        If True, enforce θ_q = θ_{n-1-q} symmetry on each layer.
    name
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed circuit with attributes:
        * ``input_params`` – ParameterVector(s) used.
        * ``num_rot_layers`` – Effective number of rotation layers.
        * ``symmetry`` – Symmetry flag.
        * ``share_params_across_layers`` – Parameter sharing flag.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    n = int(num_qubits)

    # Determine rotation layer count
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter count per layer
    param_per_layer = (n + 1) // 2 if symmetry else n

    if share_params_across_layers:
        # Single ParameterVector reused across layers
        params = ParameterVector(parameter_prefix, param_per_layer)
    else:
        # Separate parameters per layer
        params = ParameterVector(parameter_prefix, param_per_layer * num_rot_layers)

    qc = QuantumCircuit(n, name=name or "RealAmplitudesSymmetry")

    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * param_per_layer if not share_params_across_layers else 0
        for q in range(n):
            idx = min(q, n - 1 - q) if symmetry else q
            qc.ry(params[base + idx], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
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
    qc.symmetry = symmetry  # type: ignore[attr-defined]
    qc.share_params_across_layers = share_params_across_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesSymmetry(QuantumCircuit):
    """Class‑style wrapper for the symmetry‑constrained RealAmplitudes ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        share_params_across_layers: bool = False,
        symmetry: bool = True,
        name: str = "RealAmplitudesSymmetry",
    ) -> None:
        built = real_amplitudes_symmetric(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            share_params_across_layers=share_params_across_layers,
            symmetry=symmetry,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        # expose the same attributes as the functional builder
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.symmetry = built.symmetry  # type: ignore[attr-defined]
        self.share_params_across_layers = built.share_params_across_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesSymmetry", "real_amplitudes_symmetric"]
