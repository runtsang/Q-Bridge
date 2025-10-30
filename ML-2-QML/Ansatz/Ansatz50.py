"""MirroredRealAmplitudes ansatz – a symmetry‑constrained variant of RealAmplitudes."""
from __future__ import annotations

from math import ceil
from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Resolve a pairwise entanglement schedule.

    The function supports the same string identifiers as the original
    RealAmplitudes builder, plus a ``mirror`` option that entangles each qubit
    with its reflection partner ``(i, n-1-i)``.  Custom callables or explicit
    sequences are also accepted.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.

    Returns
    -------
    List[Tuple[int, int]]
        List of distinct two‑qubit pairs for CX gates.

    Raises
    ------
    ValueError
        If an unknown string is supplied or if a pair references an out‑of‑range
        qubit or a self‑pair.
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
        if entanglement == "mirror":
            return [(i, num_qubits - 1 - i) for i in range(num_qubits // 2)]
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


def mirrored_real_amplitudes(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "mirror",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build a symmetry‑constrained RealAmplitudes ansatz.

    The circuit consists of ``reps`` rotation layers followed by entangling CX
    gates, optionally ending with a final rotation layer.  Rotation parameters
    are shared across mirror qubits: the qubit pair ``(i, n-1-i)`` uses the
    same angle.  When ``num_qubits`` is odd, the central qubit receives its
    own independent parameter.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default=1
        Number of entanglement cycles.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement schedule.  ``"mirror"`` is the default.
    skip_final_rotation_layer : bool, default=False
        If ``True``, omit the final rotation layer after the last entanglement.
    insert_barriers : bool, default=False
        Insert barriers between layers for clearer circuit visualization.
    parameter_prefix : str, default="theta"
        Prefix for the ParameterVector names.
    name : str | None, default=None
        Optional circuit name; falls back to ``"MirroredRealAmplitudes"``.

    Returns
    -------
    QuantumCircuit
        A parameterized circuit that can be composed, transpiled, or executed.

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
    qc = QuantumCircuit(n, name=name or "MirroredRealAmplitudes")

    # Number of independent parameters per rotation layer
    params_per_layer = ceil(n / 2)
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * params_per_layer)

    def _rotation_layer(layer_idx: int) -> None:
        """Apply a mirror‑symmetric RY rotation to all qubits."""
        base = layer_idx * params_per_layer
        for q in range(n):
            # Mirror index: min(q, n-1-q)
            mirror_idx = min(q, n - 1 - q)
            qc.ry(params[base + mirror_idx], q)

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


class MirroredRealAmplitudes(QuantumCircuit):
    """Class‑style wrapper for a symmetry‑constrained RealAmplitudes ansatz.

    The wrapper behaves exactly like Qiskit’s ``QuantumCircuit`` while exposing
    the ``input_params`` and ``num_rot_layers`` attributes for convenient
    parameter handling.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    reps : int, default=1
        Number of entanglement cycles.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification; defaults to ``"mirror"``.
    skip_final_rotation_layer : bool, default=False
        Whether to omit the final rotation layer.
    insert_barriers : bool, default=False
        Insert barriers between layers.
    parameter_prefix : str, default="theta"
        Prefix for parameter names.
    name : str, default="MirroredRealAmplitudes"
        Optional circuit name.

    Raises
    ------
    ValueError
        Propagated from :func:`mirrored_real_amplitudes`.
    """
    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "mirror",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "MirroredRealAmplitudes",
    ) -> None:
        built = mirrored_real_amplitudes(
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


__all__ = ["MirroredRealAmplitudes", "mirrored_real_amplitudes"]
