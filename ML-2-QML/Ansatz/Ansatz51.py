"""RealAmplitudesCZControlled: a symmetry‑constrained variant of RealAmplitudesCZ."""

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


def real_amplitudes_cz_controlled(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    enforce_symmetry: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a RealAmplitudesCZ circuit with optional symmetry enforcement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of repeated rotation‑entanglement blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the two‑qubit entangling pattern.
    skip_final_rotation_layer : bool, default False
        If True, the final rotation layer is omitted.
    insert_barriers : bool, default False
        If True, insert barriers between layers for easier debugging.
    parameter_prefix : str, default "theta"
        Prefix used when generating parameter names.
    enforce_symmetry : bool, default False
        When True, the rotation parameters for qubits 0 and 1 are tied together,
        reducing the number of free parameters and enforcing an exchange symmetry.
    name : str | None, default None
        Optional name for the resulting circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if enforce_symmetry and num_qubits < 2:
        raise ValueError("enforce_symmetry requires at least 2 qubits.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZControlled")

    # Determine how many rotation layers are present.
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter vector: shared parameters for qubits 0 and 1 if symmetry is enforced.
    params_per_layer = n if not enforce_symmetry else 1 + (n - 2)
    total_params = num_rot_layers * params_per_layer
    params = ParameterVector(parameter_prefix, total_params)

    def _rot(layer: int) -> None:
        """Apply a rotation layer for the given repetition."""
        base = layer * params_per_layer
        if enforce_symmetry:
            # Shared parameter for qubits 0 and 1
            shared_param = params[base]
            qc.ry(shared_param, 0)
            qc.ry(shared_param, 1)
            # Parameters for remaining qubits
            for q in range(2, n):
                qc.ry(params[base + 1 + (q - 2)], q)
        else:
            for q in range(n):
                qc.ry(params[base + q], q)

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


class RealAmplitudesCZControlled(QuantumCircuit):
    """Convenience subclass that builds the symmetry‑controlled RealAmplitudesCZ ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        enforce_symmetry: bool = False,
        name: str = "RealAmplitudesCZControlled",
    ) -> None:
        built = real_amplitudes_cz_controlled(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            enforce_symmetry,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZControlled", "real_amplitudes_cz_controlled"]
