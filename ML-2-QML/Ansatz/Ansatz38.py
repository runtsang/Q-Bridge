"""RealAmplitudesCZExtended – a deeper, hybrid ansatz based on RealAmplitudes with CZ entanglement.

This module extends the original RealAmplitudesCZ ansatz by:
- Adding an optional universal U3 rotation layer (hybrid rotation) after each real‑amplitude layer.
- Allowing an optional second entangling layer per repetition (full connectivity).
- Providing a parity‑controlled entanglement pattern that can be interleaved.
- Exposing a convenience constructor and a subclass for easy composition.

The ansatz remains fully parameterised and compatible with Qiskit’s parameter binding and circuit composition utilities.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
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


def _parity_entanglement(num_qubits: int) -> List[Tuple[int, int]]:
    """Return pairs of qubits whose indices sum to an even number."""
    pairs: List[Tuple[int, int]] = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if (i + j) % 2 == 0:
                pairs.append((i, j))
    return pairs


# --------------------------------------------------------------------------- #
# Ansatz construction
# --------------------------------------------------------------------------- #
def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    second_entanglement: bool = False,
    parity_entanglement: bool = False,
    hybrid_rotation: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Return a Real‑Amplitude ansatz with CZ entanglement, optionally enriched with
    hybrid U3 layers, a second entangling layer, and parity‑controlled entanglement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of repetition blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement pattern for the primary CZ layer.
    skip_final_rotation_layer : bool, default False
        If True, omit the final real‑amplitude rotation layer.
    insert_barriers : bool, default False
        If True, insert barriers between layers for visual clarity.
    second_entanglement : bool, default False
        If True, add a second full‑connectivity CZ layer after the first.
    parity_entanglement : bool, default False
        If True, add a parity‑controlled CZ entanglement layer after the second.
    hybrid_rotation : bool, default False
        If True, insert a universal U3 rotation layer after each real‑amplitude layer.
    parameter_prefix : str, default "theta"
        Prefix for the parameter names.
    name : str | None, default None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    # Determine rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    real_params_count = num_rot_layers * n
    hybrid_params_count = reps * 3 * n if hybrid_rotation else 0
    total_params = real_params_count + hybrid_params_count

    params = ParameterVector(parameter_prefix, total_params)

    def _apply_real_rot(base: int) -> None:
        for q in range(n):
            qc.ry(params[base + q], q)

    def _apply_hybrid_rot(base: int) -> None:
        for q in range(n):
            idx = base + 3 * q
            qc.u3(params[idx], params[idx + 1], params[idx + 2], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        # Real‑amplitude rotation layer
        _apply_real_rot(r * n)

        # Hybrid U3 rotation layer (if requested)
        if hybrid_rotation:
            _apply_hybrid_rot(real_params_count + r * 3 * n)

        if insert_barriers:
            qc.barrier()

        # Primary CZ entanglement
        for (i, j) in pairs:
            qc.cz(i, j)

        if second_entanglement:
            # Second full‑connectivity CZ layer
            full_pairs = _resolve_entanglement(n, "full")
            for (i, j) in full_pairs:
                qc.cz(i, j)

        if parity_entanglement:
            parity_pairs = _parity_entanglement(n)
            for (i, j) in parity_pairs:
                qc.cz(i, j)

        if insert_barriers:
            qc.barrier()

    # Final rotation layer (if not skipped)
    if not skip_final_rotation_layer:
        _apply_real_rot(reps * n)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZExtended(QuantumCircuit):
    """Class wrapper for the extended CZ‑entangling Real‑Amplitudes ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        second_entanglement: bool = False,
        parity_entanglement: bool = False,
        hybrid_rotation: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            second_entanglement,
            parity_entanglement,
            hybrid_rotation,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]
