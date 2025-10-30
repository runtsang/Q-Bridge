"""RealAmplitudes variant using CZ entanglement with optional symmetry constraint on rotation parameters."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the two‑qubit entangling pattern. Accepts the strings
        ``"full"``, ``"linear"``, and ``"circular"``, a concrete sequence of
        pairs, or a callable that returns such a sequence.

    Returns
    -------
    List[Tuple[int, int]]
        List of distinct qubit pairs to be entangled.

    Raises
    ------
    ValueError
        If an unknown string is supplied or if a pair is invalid.
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


def real_amplitudes_cz_controlled_modification(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    symmetry: bool = True,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a RealAmplitudes circuit using CZ entanglement with an optional
    mirror‑symmetry constraint on the rotation parameters.

    The circuit consists of ``reps`` rotation layers followed by entangling CZ
    gates.  An additional rotation layer is appended unless
    ``skip_final_rotation_layer`` is ``True``.  When ``symmetry`` is ``True``
    (default), the rotation parameters are constrained to satisfy
    ``theta_i = theta_{n-1-i}`` for each layer, effectively halving the number
    of independent parameters per layer.  The function preserves all optional
    features of the original RealAmplitudesCZ ansatz, such as custom
    entanglement schedules and barrier insertion.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of repetition blocks (rotation + entanglement).
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern.  See ``_resolve_entanglement`` for details.
    skip_final_rotation_layer : bool, default False
        If ``True``, omit the final rotation layer that normally follows the
        last entanglement block.
    insert_barriers : bool, default False
        If ``True``, insert a barrier after each rotation and entanglement
        block to aid circuit visualization and debugging.
    parameter_prefix : str, default "theta"
        Prefix for the parameter vector names.
    symmetry : bool, default True
        Enforce mirror symmetry on the rotation parameters across qubits.
    name : str | None, default None
        Name of the resulting circuit.  If ``None``, a default name is
        constructed from the class name.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1 or if ``reps`` is negative.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZControlled")

    # Determine number of rotation layers (including optional final layer)
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter vector length depends on symmetry setting
    half_params = (n + 1) // 2  # ceil(n/2)
    if symmetry:
        total_params = num_rot_layers * half_params
    else:
        total_params = num_rot_layers * n
    params = ParameterVector(parameter_prefix, total_params)

    def _rotation_layer(layer: int) -> None:
        """Apply the rotation layer for a given repetition index."""
        base = layer * (half_params if symmetry else n)
        for q in range(n):
            if symmetry:
                # Mirror symmetry: use param for q and its counterpart
                idx = base + (q if q < half_params else n - 1 - q)
            else:
                idx = base + q
            qc.ry(params[idx], q)

    ent_pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        for i, j in ent_pairs:
            qc.cz(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    # Attach metadata for downstream tooling
    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    qc.symmetry = symmetry  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZControlledModification(QuantumCircuit):
    """Convenience wrapper that builds a RealAmplitudes circuit with CZ
    entanglement and optional symmetry constraint.

    The constructor simply forwards all arguments to
    :func:`real_amplitudes_cz_controlled_modification` and composes the result
    into the current circuit.  The resulting object exposes the same public
    attributes as the underlying circuit, including ``input_params`` and
    ``num_rot_layers``.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        symmetry: bool = True,
        name: str = "RealAmplitudesCZControlled",
    ) -> None:
        built = real_amplitudes_cz_controlled_modification(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            symmetry,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.symmetry = built.symmetry  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZControlledModification", "real_amplitudes_cz_controlled_modification"]
