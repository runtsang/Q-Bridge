"""Symmetric Real Amplitudes ansatz with mirrored parameter sharing."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Resolve entanglement specification into a list of qubit pairs."""
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
    name: str | None = None,
) -> QuantumCircuit:
    """Build a symmetric RealAmplitudes-style circuit with mirrored parameter sharing.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit. Must be >= 1.
    reps : int, default 1
        Number of entanglement blocks. The total number of rotation layers is
        ``reps`` if ``skip_final_rotation_layer`` is True, otherwise ``reps + 1``.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Specification of the two‑qubit entanglement pattern.
    skip_final_rotation_layer : bool, default False
        If True, the circuit ends after the last entanglement block without a
        final rotation layer.
    insert_barriers : bool, default False
        Insert barriers between rotation and entanglement blocks for visual
        clarity.
    parameter_prefix : str, default "theta"
        Prefix for the parameter vector naming.
    name : str | None, default None
        Circuit name. If None, defaults to ``"RealAmplitudesSymmetric"``.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesSymmetric")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params_per_layer = (n + 1) // 2  # mirror symmetry halves the parameter count
    params = ParameterVector(parameter_prefix, num_rot_layers * params_per_layer)

    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * params_per_layer
        for q in range(n):
            sym_idx = min(q, n - 1 - q)  # mirror index
            qc.ry(params[base + sym_idx], q)

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
    return qc


class RealAmplitudesSymmetric(QuantumCircuit):
    """QuantumCircuit subclass implementing the symmetric RealAmplitudes ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesSymmetric",
    ) -> None:
        built = real_amplitudes_symmetric(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesSymmetric", "real_amplitudes_symmetric"]
