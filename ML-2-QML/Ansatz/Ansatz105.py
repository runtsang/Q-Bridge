"""RealAmplitudes variant with hybrid CZ/iSWAP entanglers for deeper expressivity.

The module defines a new ansatz class and a convenience function.
The design expands the ansatz without altering the core rotation‑layer logic.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# --------------------------------------------------------------------------- #
# Helper: Entanglement resolution
# --------------------------------------------------------------------------- #
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Specification of which qubit pairs to entangle.

    Returns
    -------
    List[Tuple[int, int]]
        List of distinct qubit pairs.
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
        return [(int(i), int(j)) for i, j in pairs]

    pairs = [(int(i), int(j)) for i, j in entanglement]
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
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
    parameter_prefix: str = "theta",
    name: str | None = None,
    hybrid_depth: int = 1,
    deep_copy_entanglement: bool = True,
) -> QuantumCircuit:
    """
    Construct a CZ‑based RealAmplitudes ansatz with an optional hybrid CZ/iSWAP entanglement layer.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of rotation–entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable, default "full"
        Specification of entanglement pairs.
    skip_final_rotation_layer : bool, default False
        If ``True``, omit the final rotation layer.
    insert_barriers : bool, default False
        Insert barriers between logical layers for easier circuit inspection.
    parameter_prefix : str, default "theta"
        Prefix used for the rotation parameters.
    name : str, optional
        Name of the resulting circuit.
    hybrid_depth : int, default 1
        Number of hybrid entanglement cycles inserted between each standard rotation block.
    deep_copy_entanglement : bool, default True
        If ``True``, make a defensive copy of the entanglement pair list.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Notes
    -----
    * Each rotation layer applies `ry` gates with independent parameters.
    * Between rotation layers, a standard CZ entanglement layer is applied.
    * A *hybrid* entanglement layer follows, alternating between CZ and iSWAP gates
      for successive cycles.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if hybrid_depth < 0:
        raise ValueError("hybrid_depth must be non‑negative.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCzExtended")

    # Determine number of rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        base = layer * n
        for q in range(n):
            qc.ry(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)
    if deep_copy_entanglement:
        pairs = [tuple(p) for p in pairs]

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for i, j in pairs:
            qc.cz(i, j)
        if insert_barriers:
            qc.barrier()

        # Hybrid entanglement cycles
        for h in range(hybrid_depth):
            if insert_barriers:
                qc.barrier()
            for i, j in pairs:
                if h % 2 == 0:
                    qc.cz(i, j)
                else:
                    qc.iswap(i, j)
            if insert_barriers:
                qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


# --------------------------------------------------------------------------- #
# Class wrapper
# --------------------------------------------------------------------------- #
class RealAmplitudesCzExtended(QuantumCircuit):
    """Convenience wrapper for the CZ‑based RealAmplitudes ansatz with hybrid entanglement.

    Parameters are identical to :func:`real_amplitudes_cz_extended`.  The class
    simply builds the circuit via the helper function and composes it into the
    current instance, exposing the same ``input_params`` and ``num_rot_layers``
    attributes for downstream tooling.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCzExtended",
        hybrid_depth: int = 1,
        deep_copy_entanglement: bool = True,
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
            hybrid_depth,
            deep_copy_entanglement,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCzExtended", "real_amplitudes_cz_extended"]
