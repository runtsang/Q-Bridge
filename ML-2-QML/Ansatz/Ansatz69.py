"""
RealAmplitudesAlternatingExtended: a depth‑enhanced, Qiskit‑compatible ansatz.

This module extends the original RealAmplitudesAlternating by adding a second
rotation block per repetition and an optional controlled‑phase entanglement
stage.  The new ansatz keeps the alternating‑rotation intuition but offers
greater expressivity and a richer entanglement schedule.

Key extensions
---------------
- **Double rotation layers**: each repetition now applies two distinct
  rotation blocks.  The first block alternates RY / RX as in the seed, while
  the second block alternates RZ / RY.  This provides a richer single‑qubit
  variational space.

- **Optional controlled‑phase (CPHASE) block**: after the entangling CX gates
  a user‑controlled CPHASE stage can be inserted.  One parameter per pair is
  used, allowing the ansatz to explore long‑range phase correlations.

- **Flexible depth control**: the `reps` argument controls the number of
  repetition cycles; the `skip_final_rotation_layer` flag mirrors the seed
  behaviour.  The total number of rotation layers is `2*reps + (0 or 1)`.

- **Barrier support**: the `insert_barriers` flag can be used to aid
  visualisation or debugging.

All parameters are exposed through a single ParameterVector named
`theta`.  If the phase block is enabled, a second vector named `phi` is
created and attached to the circuit as `phase_params`.

Usage
-----
```python
from ansatz_scaled.real_amplitudes_alternating_extension import (
    RealAmplitudesAlternatingExtended,
    real_amplitudes_alternating_extended,
)

# Build a 3‑qubit circuit with 2 repetitions and phase entanglement
qc = real_amplitudes_alternating_extended(3, reps=2, phase_entanglement=True)
```
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement_pairs(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Translate a user‑supplied entanglement specification into a list of
    qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Either a keyword string ('full', 'linear', 'circular') or an explicit
        sequence of pairs.  A callable is also accepted for dynamic
        construction.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of distinct qubit pairs.

    Raises
    ------
    ValueError
        If an unknown keyword is supplied or a pair references an
        out‑of‑range qubit.
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


def real_amplitudes_alternating_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    phase_entanglement: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Construct a depth‑enhanced alternating‑rotation ansatz.

    The circuit alternates between RY/RX and RZ/RY rotation blocks for each
    repetition.  After each pair of rotations a CX entangling layer is
    applied, optionally followed by a CPHASE layer if `phase_entanglement`
    is True.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of repetition cycles.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement pattern for the CX gates.
    skip_final_rotation_layer : bool, default False
        If True, the circuit ends after the last CX layer without an
        additional rotation block.
    insert_barriers : bool, default False
        Insert barriers between logical blocks for easier visualisation.
    phase_entanglement : bool, default False
        Add a controlled‑phase (CPHASE) gate between the same qubit pairs
        used for the CX gates.  One parameter per pair is created.
    parameter_prefix : str, default "theta"
        Prefix for the rotation parameters.
    name : str | None, default None
        Circuit name.  Defaults to ``"RealAmplitudesAlternatingExtended"``.

    Returns
    -------
    QuantumCircuit
        The constructed variational circuit.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingExtended")

    # Determine number of rotation layers
    num_rot_layers = 2 * reps + (0 if skip_final_rotation_layer else 1)
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    # Optional phase parameters
    phase_params: ParameterVector | None = None
    if phase_entanglement:
        pairs = _resolve_entanglement_pairs(n, entanglement)
        phase_params = ParameterVector("phi", len(pairs))

    def _apply_rotation(layer: int, base: int) -> None:
        """Apply a rotation block.  Even layers use RY / RX, odd layers use RZ / RY."""
        if layer % 2 == 0:
            for q in range(n):
                qc.ry(params[base + q], q)
        else:
            for q in range(n):
                qc.rz(params[base + q], q)
            for q in range(n):
                qc.ry(params[base + q + n], q)

    # Build the circuit
    pairs = _resolve_entanglement_pairs(n, entanglement)
    for r in range(reps):
        # First rotation block
        base = (2 * r) * n
        _apply_rotation(2 * r, base)
        if insert_barriers:
            qc.barrier()

        # CX entanglement
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

        # Second rotation block
        base = (2 * r + 1) * n
        _apply_rotation(2 * r + 1, base)
        if insert_barriers:
            qc.barrier()

        # Optional phase entanglement
        if phase_entanglement:
            for idx, (i, j) in enumerate(pairs):
                qc.cphase(phase_params[idx], i, j)
            if insert_barriers:
                qc.barrier()

    # Final rotation block if requested
    if not skip_final_rotation_layer:
        base = (num_rot_layers - 1) * n
        _apply_rotation(num_rot_layers - 1, base)

    qc.input_params = params
    qc.num_rot_layers = num_rot_layers
    if phase_entanglement:
        qc.phase_params = phase_params  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingExtended(QuantumCircuit):
    """Convenience wrapper that builds a depth‑enhanced alternating‑rotation ansatz.

    Parameters are identical to :func:`real_amplitudes_alternating_extended` and
    are forwarded directly to the underlying construction function.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        phase_entanglement: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingExtended",
    ) -> None:
        built = real_amplitudes_alternating_extended(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            phase_entanglement,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        if phase_entanglement:
            self.phase_params = built.phase_params  # type: ignore[attr-defined]


__all__ = [
    "RealAmplitudesAlternatingExtended",
    "real_amplitudes_alternating_extended",
]
