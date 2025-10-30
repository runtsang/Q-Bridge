"""RealAmplitudesCZSymmetry:  a symmetry‑constrained variant of RealAmplitudes with CZ entanglement.

This ansatz enforces that every rotation layer applies a *single* angle to all qubits.
The result is a circuit that respects the global‑Z symmetry (Z⊗n) while retaining
the expressive power of the original RealAmplitudesCZ circuit.  The number of
parameters is reduced to :math:`\\text{num_rot_layers}`.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Resolve the entanglement specification into a list of qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern; can be a string shortcut, a list of pairs, or a callable that
        generates the pairs.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs for CZ gates.

    Raises
    ------
    ValueError
        If the specification is invalid or contains out‑of‑range indices.
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


def real_amplitudes_cz_symmetry(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build a symmetry‑constrained RealAmplitudes circuit with CZ gates.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of repetition blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement pattern for the CZ gates.
    skip_final_rotation_layer : bool, default False
        Whether to omit the final rotation layer.
    insert_barriers : bool, default False
        Whether to insert barriers between layers for clarity.
    parameter_prefix : str, default "theta"
        Prefix for the parameter names.
    name : str | None, default None
        Name of the resulting circuit.

    Returns
    -------
    QuantumCircuit
        A parameterized quantum circuit with symmetry‑constrained rotations.

    Notes
    -----
    * Each rotation layer uses a single parameter shared across all qubits.
    * The circuit is equivalent to the original RealAmplitudesCZ when the number of
      parameters per layer equals the number of qubits, but it enforces a global‑Z
      symmetry.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZSymmetry")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    # One parameter per rotation layer
    params = ParameterVector(parameter_prefix, num_rot_layers)

    def _rot(layer: int) -> None:
        """Apply a shared rotation to all qubits in the given layer."""
        angle = params[layer]
        for q in range(n):
            qc.ry(angle, q)

    pairs = _resolve_entanglement(n, entanglement)

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
    """Convenience subclass that constructs the symmetry‑constrained ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of repetition blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement pattern for the CZ gates.
    skip_final_rotation_layer : bool, default False
        Whether to omit the final rotation layer.
    insert_barriers : bool, default False
        Whether to insert barriers between layers for clarity.
    parameter_prefix : str, default "theta"
        Prefix for the parameter names.
    name : str, default "RealAmplitudesCZSymmetry"
        Name of the quantum circuit.

    Notes
    -----
    The subclass simply wraps :func:`real_amplitudes_cz_symmetry` and exposes the
    resulting parameters via ``input_params`` and ``num_rot_layers`` attributes.
    """
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


__all__ = ["RealAmplitudesCZSymmetry", "real_amplitudes_cz_symmetry"]
