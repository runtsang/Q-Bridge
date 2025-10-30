"""RealAmplitudes variant with alternating RY/RX rotation layers and enforced mirror symmetry."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple
import math

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str or sequence or callable
        Specification of entanglement:
        ``"full"``   – all pairs,
        ``"linear"`` – nearest‑neighbour chain,
        ``"circular"`` – chain with a closing CX,
        or a custom sequence/callable returning pairs.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of (control, target) pairs.

    Raises
    ------
    ValueError
        If an unknown string is supplied or a pair is invalid.
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


def real_amplitudes_alternating_controlled_modification(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    r"""Create a RealAmplitudes ansatz with alternating RY/RX layers and mirror symmetry.

    The circuit alternates between :class:`~qiskit.circuit.library.RY` and
    :class:`~qiskit.circuit.library.RX` rotations on each qubit.  Unlike the
    vanilla ``RealAmplitudesAlternating`` ansatz, this variant enforces that
    qubits ``q`` and ``n-1-q`` share the same rotation angle in every layer.
    Consequently the number of independent parameters is reduced to
    ``ceil(n/2)`` per rotation layer.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit. Must be >= 1.
    reps : int, default 1
        Number of entanglement blocks. The total number of rotation layers is
        ``reps`` if ``skip_final_rotation_layer`` is ``True``, otherwise
        ``reps + 1``.
    entanglement : str or sequence or callable, default "full"
        Entanglement pattern. See :func:`_resolve_entanglement` for details.
    skip_final_rotation_layer : bool, default False
        If ``True`` the final rotation layer after the last entanglement block
        is omitted.
    insert_barriers : bool, default False
        Whether to insert barriers between logical blocks for easier
        visualisation or debugging.
    parameter_prefix : str, default "theta"
        Prefix used for the :class:`~qiskit.circuit.ParameterVector`.
    name : str, optional
        Name for the resulting :class:`~qiskit.QuantumCircuit`. If ``None`` a
        sensible default is used.

    Returns
    -------
    QuantumCircuit
        The constructed circuit.  The circuit exposes two additional
        attributes:
        ``input_params`` – the :class:`~qiskit.circuit.ParameterVector`
        containing all free parameters.
        ``num_rot_layers`` – the number of rotation layers added.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingControlled")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    unique_params_per_layer = math.ceil(n / 2)
    params = ParameterVector(parameter_prefix, num_rot_layers * unique_params_per_layer)

    def _rot(layer: int) -> None:
        base = layer * unique_params_per_layer
        for q in range(n):
            # Mirror index: the same angle is used for qubit q and its partner n-1-q
            idx = base + min(q, n - 1 - q)
            if layer % 2 == 0:
                qc.ry(params[idx], q)
            else:
                qc.rx(params[idx], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingControlledModification(QuantumCircuit):
    """Convenience subclass for the symmetric alternating‑rotation ansatz.

    The class simply constructs the circuit via
    :func:`real_amplitudes_alternating_controlled_modification` and
    exposes the same public attributes (``input_params`` and ``num_rot_layers``).
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingControlled",
    ) -> None:
        built = real_amplitudes_alternating_controlled_modification(
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


__all__ = [
    "RealAmplitudesAlternatingControlledModification",
    "real_amplitudes_alternating_controlled_modification",
]
