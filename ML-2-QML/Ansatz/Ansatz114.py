"""RealAmplitudesCZ Controlled variant with shared rotation angles."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = [
    "RealAmplitudesCZControlled",
    "real_amplitudes_cz_controlled",
]


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Return a list of two‑qubit pairs according to a simple entanglement spec.

    Parameters
    ----------
    num_qubits :
        Number of qubits in the circuit.
    entanglement :
        Either a string keyword ("full", "linear", "circular"),
        a user‑supplied sequence of pairs, or a callable that
        returns a sequence of pairs given ``num_qubits``.
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
            raise ValueError(
                f"Entanglement pair {(i, j)} out of range for n={num_qubits}."
            )
    return pairs


def real_amplitudes_cz_controlled(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a CZ‑entangling Real‑Amplitudes ansatz with *shared* rotation angles per layer.

    The ansatz is a controlled modification of the original ``real_amplitudes_cz``:
    instead of allocating ``n`` independent rotation parameters for each layer,
    a *single* parameter is reused on every qubit within that layer.
    This symmetry preserves the circuit topology while reducing the parameter count.

    Parameters
    ----------
    num_qubits :
        Number of qubits in the ansatz.
    reps :
        Number of entanglement–rotation repetitions.
    entanglement :
        Specification of qubit pairs to entangle (see ``_resolve_entanglement``).
    skip_final_rotation_layer :
        If ``True`` no rotation layer is applied after the last entanglement block.
    insert_barriers :
        Insert a barrier after each rotation layer and after each entanglement block.
    parameter_prefix :
        Prefix for the ``ParameterVector`` names.
    name :
        Optional name for the resulting circuit.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZControlled")

    # Number of rotation layers = reps if final layer is skipped, otherwise reps+1
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers)

    def _rot(layer: int) -> None:
        """Apply a shared Ry rotation to all qubits in ``layer``."""
        for q in range(n):
            qc.ry(params[layer], q)

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
    """Convenience subclass for the shared‑parameter CZ Real‑Amplitudes ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZControlled",
    ) -> None:
        built = real_amplitudes_cz_controlled(
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
