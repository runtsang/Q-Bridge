"""RealAmplitudesCZSymmetry ansatz.

This module implements a variant of the RealAmplitudes ansatz that uses CZ
entanglers and enforces two symmetry constraints:
1. All qubits share a single rotation angle per layer (parameter sharing).
2. The entanglement pattern can be reversed on every other layer (mirror
   symmetry) to increase expressivity without changing the gate set.

The implementation follows the same API as the original `real_amplitudes_cz`
function: ``real_amplitudes_cz_symmetry`` and its class wrapper
``RealAmplitudesCZSymmetry``.  It is fully Qiskit compatible – the circuit
can be composed, parameters can be bound, and it exposes ``input_params``
and ``num_rot_layers`` attributes for introspection.

Typical usage:

>>> from qiskit import QuantumCircuit
>>> qc = real_amplitudes_cz_symmetry(4, reps=2, reverse_entanglement=True)
>>> qc.draw()
"""

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


def real_amplitudes_cz_symmetry(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    reverse_entanglement: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a parameter‑shared RealAmplitudes ansatz with CZ entanglers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of entangling‑rotation blocks.
    entanglement : str or sequence or callable, default ``"full"``
        Specification of the two‑qubit entanglement pattern.
    skip_final_rotation_layer : bool, default False
        If True, the final rotation layer is omitted.
    insert_barriers : bool, default False
        Whether to insert barriers between layers for visual clarity.
    reverse_entanglement : bool, default False
        If True, the entanglement pair list is reversed on every odd layer.
    parameter_prefix : str, default ``"theta"``
        Prefix for the ParameterVector names.
    name : str, optional
        Circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.  The circuit exposes the attributes
        ``input_params`` (``ParameterVector``) and ``num_rot_layers`` (int)
        for convenient introspection.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1 or if ``reps`` is negative.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    if reps < 0:
        raise ValueError("reps must be >= 0.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZSymmetry")

    # Number of rotation layers: one per repetition, plus an optional final layer.
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers)

    def _rot(layer: int) -> None:
        """Apply a shared Ry rotation to all qubits for the given layer."""
        angle = params[layer]
        for q in range(n):
            qc.ry(angle, q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        # Optionally reverse the pair list on odd layers to introduce mirror symmetry.
        current_pairs = list(reversed(pairs)) if reverse_entanglement and (r % 2 == 1) else pairs
        for (i, j) in current_pairs:
            qc.cz(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZSymmetry(QuantumCircuit):
    """Convenience subclass for the parameter‑shared RealAmplitudes CZ ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        reverse_entanglement: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZSymmetry",
    ) -> None:
        built = real_amplitudes_cz_symmetry(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            reverse_entanglement,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZSymmetry", "real_amplitudes_cz_symmetry"]
