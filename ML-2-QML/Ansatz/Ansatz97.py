"""RealAmplitudesCZExtended: a depth‑controlled, hybrid‑rotation ansatz with CZ entanglement.

This module extends the original `RealAmplitudesCZ` by adding
additional hybrid rotation‑entanglement layers. The new `depth`
parameter controls how many extra layers are appended after the
original circuit. Each extra layer consists of a full‑connect
CZ entanglement block followed by a rotation layer that acts on all
qubits. The design keeps the original interface intact, making
the new ansatz drop‑in compatible with existing code.

Key design points
-----------------
* **Depth control** – `depth` adds extra layers without affecting
  the core structure.
* **Hybrid rotation** – extra layers use a global rotation (all
  qubits), which increases expressivity.
* **Parameter sharing** – each rotation layer gets its own set of
  parameters; parameters are not shared across layers.
* **Barrier insertion** – optional barriers can be inserted between
  logical blocks for debugging or for hardware with restricted
  connectivity.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement_pairs(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement specification.

    Parameters
    ----------
    num_qubits : int
        The number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]]
        A string description ('full', 'linear', 'circular') or an explicit list or
        a callable that generates the list of pairs.

    Returns
    -------
    List[Tuple[int, int]]
        A validated list of distinct pairs.

    Raises
    ------
    ValueError
        If an unknown string is provided or if any pair is invalid.
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


def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
    depth: int = 0,
) -> QuantumCircuit:
    """Build a depth‑controlled, hybrid‑rotation RealAmplitudes ansatz with CZ entanglement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int
        Number of original rotation–entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]]
        Entanglement specification for the original layers.
    skip_final_rotation_layer : bool, optional
        If ``True``, omit the final rotation layer that normally follows the last
        entanglement block.
    insert_barriers : bool, optional
        Insert barriers between logical blocks for easier debugging.
    parameter_prefix : str, optional
        Prefix used for parameter names.
    name : str | None, optional
        Name of the resulting quantum circuit.
    depth : int, optional
        Number of additional hybrid layers appended after the original
        circuit. Each additional layer contains a full‑connect CZ block
        followed by a rotation layer acting on all qubits.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Raises
    ------
    ValueError
        If ``num_qubits`` or ``depth`` is negative.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if depth < 0:
        raise ValueError("depth must be >= 0.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    # Compute total number of rotation layers
    base_rot_layers = reps if skip_final_rotation_layer else reps + 1
    total_rot_layers = base_rot_layers + depth
    params = ParameterVector(parameter_prefix, total_rot_layers * n)

    def _rotation_layer(layer: int) -> None:
        """Apply a Y‑rotation on each qubit for the specified layer."""
        base = layer * n
        for q in range(n):
            qc.ry(params[base + q], q)

    # Resolve entanglement pairs for the original circuit
    pairs = _resolve_entanglement_pairs(n, entanglement)

    # Build the original circuit
    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cz(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    # Append the extra hybrid layers
    for e in range(depth):
        _rotation_layer(base_rot_layers + e)
        if insert_barriers:
            qc.barrier()
        # Full‑connect CZ for the extra layer
        full_pairs = _resolve_entanglement_pairs(n, "full")
        for (i, j) in full_pairs:
            qc.cz(i, j)
        if insert_barriers:
            qc.barrier()

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = total_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZExtended(QuantumCircuit):
    """Convenience wrapper for the depth‑controlled RealAmplitudes‑CZ ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, optional
        Number of original rotation–entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]], optional
        Entanglement specification for the original layers.
    skip_final_rotation_layer : bool, optional
        If ``True``, omit the final rotation layer.
    insert_barriers : bool, optional
        Insert barriers between logical blocks.
    parameter_prefix : str, optional
        Prefix used for parameter names.
    name : str, optional
        Name of the resulting quantum circuit.
    depth : int, optional
        Number of additional hybrid layers appended after the original circuit.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
        depth: int = 0,
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
            depth,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]
