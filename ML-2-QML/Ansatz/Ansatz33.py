"""RealAmplitudes variant with alternating RY/RX rotation layers and optional mirror symmetry."""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs based on a simple specification.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.  ``"full"``, ``"linear"``, or ``"circular"`` strings are
        accepted, or an explicit list of pairs, or a callable that returns a list for a
        given ``num_qubits``.  The function validates that pairs are distinct and within
        bounds.

    Returns
    -------
    List[Tuple[int, int]]
        List of qubit pairs to be CX‑entangled in each rotation layer.
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


def real_amplitudes_alternating_controlled(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    mirror_symmetry: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build a RealAmplitudes circuit with alternating RY/RX rotations and optional mirror symmetry.

    The circuit alternates between RY and RX rotation layers on each qubit.  An optional
    ``mirror_symmetry`` flag enforces that qubits symmetric about the circuit center
    share identical rotation angles, reducing the number of free parameters by roughly
    half.  The entanglement schedule is identical to the original ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default=1
        Number of entanglement layers.  If ``skip_final_rotation_layer`` is ``False``,
        an additional rotation layer is appended after the last entanglement step.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.  See ``_resolve_entanglement`` for details.
    skip_final_rotation_layer : bool, default=False
        If ``True``, no rotation layer is added after the last entanglement step.
    insert_barriers : bool, default=False
        If ``True``, a barrier is inserted after each rotation and entanglement block.
    mirror_symmetry : bool, default=False
        If ``True``, enforce mirror symmetry across the circuit center: qubits
        ``i`` and ``n-1-i`` share the same rotation angle in every layer.
    parameter_prefix : str, default="theta"
        Prefix for parameter names.
    name : str | None, default=None
        Name for the quantum circuit.  If ``None``, a default name is used.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit with the specified structure.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingControlled")

    # Determine number of rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter vector length depends on symmetry choice
    params_per_layer = (n + 1) // 2 if mirror_symmetry else n
    params = ParameterVector(parameter_prefix, num_rot_layers * params_per_layer)

    def _apply_rotations(layer: int) -> None:
        """Apply the alternating rotation layer to all qubits."""
        base = layer * params_per_layer
        for q in range(n):
            if mirror_symmetry:
                idx = base + min(q, n - 1 - q)
            else:
                idx = base + q
            if layer % 2 == 0:
                qc.ry(params[idx], q)
            else:
                qc.rx(params[idx], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _apply_rotations(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _apply_rotations(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingControlled(QuantumCircuit):
    """Convenience wrapper for the mirrored‑symmetry RealAmplitudes ansatz.

    The class behaves like a normal :class:`~qiskit.circuit.QuantumCircuit` but
    exposes the same configuration knobs as :func:`real_amplitudes_alternating_controlled`.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        mirror_symmetry: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingControlled",
    ) -> None:
        built = real_amplitudes_alternating_controlled(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            mirror_symmetry,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesAlternatingControlled", "real_amplitudes_alternating_controlled"]
