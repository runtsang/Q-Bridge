"""RealAmplitudes variant with a controlled global Z‑phase symmetry."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PhaseGate


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
    name: str | None = None,
) -> QuantumCircuit:
    """Construct a Real‑Amplitudes circuit using CZ entanglers with a global Z‑phase symmetry.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default=1
        Number of rotation‑entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default="full"
        Specification of which qubit pairs receive a CZ gate.
    skip_final_rotation_layer : bool, default=False
        If True, omits the last rotation layer that follows the final entanglement.
    insert_barriers : bool, default=False
        If True, inserts a barrier after each rotation and entanglement block for clarity.
    parameter_prefix : str, default="theta"
        Prefix for the parameters in the ParameterVector.
    name : str | None, default=None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Notes
    -----
    After each entangling layer a Phase(π) gate is applied to every qubit.  This
    imposes a trivial global phase symmetry while keeping the seed’s rotation
    and entanglement pattern unchanged.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZControlled")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        base = layer * n
        for q in range(n):
            qc.ry(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cz(i, j)
        # Global Z‑phase symmetry: apply Phase(π) to each qubit
        for q in range(n):
            qc.append(PhaseGate(np.pi), [q])
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZControlled(QuantumCircuit):
    """Class wrapper for the CZ‑entangling variant with a global Z‑phase symmetry."""

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


__all__ = ["RealAmplitudesCZControlled", "real_amplitudes_cz_controlled"]
