"""RealAmplitudes CZ ansatz with parity‑symmetric parameter sharing."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = ["RealAmplitudesCZSymmetry", "real_amplitudes_cz_symmetry"]

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

def real_amplitudes_cz_symmetry(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a RealAmplitudes circuit with CZ entanglers and parity‑symmetric
    rotation parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit. Must be >= 1.
    reps : int, optional
        Number of repeated rotation‑entanglement layers. Default is 1.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], optional
        Specification of the entanglement graph. Supports the same options as the
        original RealAmplitudes implementation.
    skip_final_rotation_layer : bool, optional
        If True, the final rotation layer is omitted. Default is False.
    insert_barriers : bool, optional
        If True, barriers are inserted between layers for clarity. Default is False.
    parameter_prefix : str, optional
        Prefix for the rotation parameters. Default is "theta".
    name : str | None, optional
        Optional circuit name. If None, defaults to ``"RealAmplitudesCZSymmetry"``.

    Returns
    -------
    QuantumCircuit
        A parameterized quantum circuit with symmetric rotation angles and mirrored
        CZ entanglement pairs.

    Notes
    -----
    * The number of unique rotation parameters per layer is ceil(num_qubits / 2),
      halving the parameter count compared to the unconstrained ansatz.
    * Entanglement pairs are mirrored: for each pair (i, j) also (n‑1‑i, n‑1‑j) is
      added if not already present, ensuring the entanglement graph is symmetric.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZSymmetry")

    # Determine unique parameters per layer (mirror symmetry)
    unique_params_per_layer = (n + 1) // 2
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    total_params = unique_params_per_layer * num_rot_layers
    params = ParameterVector(parameter_prefix, total_params)

    def _rot(layer: int) -> None:
        """Apply a symmetric rotation layer."""
        base = layer * unique_params_per_layer
        for q in range(n):
            idx = q if q < n // 2 else n - 1 - q
            qc.ry(params[base + idx], q)

    # Resolve entanglement and mirror it
    pairs = _resolve_entanglement(n, entanglement)
    pairs_set = set(pairs)
    for i, j in pairs:
        mi, mj = n - 1 - i, n - 1 - j
        if mi > mj:
            mi, mj = mj, mi
        pairs_set.add((mi, mj))
    pairs = sorted(pairs_set)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cz(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZSymmetry(QuantumCircuit):
    """Class wrapper for the CZ‑entangling, parity‑symmetric RealAmplitudes ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZSymmetry",
    ) -> None:
        built = real_amplitudes_cz_symmetry(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
