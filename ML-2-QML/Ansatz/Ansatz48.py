"""RealAmplitudesCZShared: Parameter‑shared variant of RealAmplitudes with CZ entanglement.

This module implements a controlled modification of the standard RealAmplitudes
ansatz.  The key change is that all rotation layers share a common set of
parameters per qubit, dramatically reducing the total number of free degrees of
freedom.  The structure of the circuit (rotation layers followed by CZ
entanglers) is preserved, and the module exposes both a convenience constructor
function and a subclass of :class:`qiskit.QuantumCircuit` with the same name.
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


def real_amplitudes_cz_shared(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    parameter_sharing: bool = True,
    name: str | None = None,
) -> QuantumCircuit:
    """Construct a RealAmplitudes ansatz with CZ entanglers and optional parameter sharing.

    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit.
    reps
        Number of repeated rotation‑entanglement layers.
    entanglement
        Specification of qubit pairs to entangle.  Can be a string
        ('full', 'linear', 'circular'), a sequence of tuples, or a callable
        that returns such a sequence given ``num_qubits``.
    skip_final_rotation_layer
        If ``True``, omit the final rotation layer after the last entanglement
        step.  Defaults to ``False``.
    insert_barriers
        If ``True``, insert barriers between layers for readability.
    parameter_prefix
        Prefix for the :class:`~qiskit.circuit.Parameter` names.
    parameter_sharing
        If ``True``, use a single parameter per qubit reused across all
        rotation layers.  If ``False``, allocate a full set of parameters per
        layer (original behaviour).
    name
        Optional circuit name.  Defaults to ``"RealAmplitudesCZShared"``.

    Returns
    -------
    QuantumCircuit
        The constructed circuit.  The attributes ``input_params`` and
        ``num_rot_layers`` are attached for convenience.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1 or ``reps`` is negative.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZShared")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    if parameter_sharing:
        params = ParameterVector(parameter_prefix, n)
    else:
        params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        """Apply a rotation layer.  Parameters may be shared across layers."""
        if parameter_sharing:
            for q in range(n):
                qc.ry(params[q], q)
        else:
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
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZShared(QuantumCircuit):
    """Class wrapper for the parameter‑shared CZ‑entangling variant of RealAmplitudes."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        parameter_sharing: bool = True,
        name: str = "RealAmplitudesCZShared",
    ) -> None:
        built = real_amplitudes_cz_shared(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            parameter_sharing,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZShared", "real_amplitudes_cz_shared"]
