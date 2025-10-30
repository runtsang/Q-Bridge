"""RealAmplitudesCZExtended – a depth‑controlled, multi‑qubit entanglement variant of RealAmplitudesCZ."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs for the base entanglement schedule.

    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit.
    entanglement
        Either a string specifier ('full', 'linear', 'circular') or a custom
        sequence/callable returning pairs of qubit indices.

    Returns
    -------
    List[Tuple[int, int]]
        Valid, validated pairs of qubits to be entangled.

    Raises
    ------
    ValueError
        If an invalid specification or out‑of‑range pair is provided.
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
    depth: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    star_entanglement: bool = False,
    share_parameters: bool = True,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Construct a depth‑controlled, parameter‑shared RealAmplitudesCZ ansatz.

    The circuit is built from a sequence of rotation layers (Ry) and
    CZ entanglement layers.  An optional *star* entanglement connects
    qubit 0 to all other qubits after each CZ layer.

    Parameters
    ----------
    num_qubits
        Number of qubits in the ansatz.
    reps
        Number of rotation/entanglement repetitions per depth slice.
    depth
        Number of times the core pattern (reps × rotation + entanglement)
        is repeated.  ``depth`` must be >= 1.
    entanglement
        Specification of CZ pairs.  See :func:`_resolve_entanglement`.
    skip_final_rotation_layer
        If ``False`` (default) an additional rotation layer is applied
        after the final repetition.
    insert_barriers
        If ``True`` barriers are inserted after each rotation and
        entanglement layer for visual clarity.
    star_entanglement
        If ``True``, after each CZ entanglement layer a star‑shaped
        CZ network (qubit 0 entangled with all others) is applied.
    share_parameters
        If ``True`` (default) the same set of parameters is reused
        across all depth slices.  If ``False`` each depth slice receives
        its own distinct parameters.
    parameter_prefix
        Prefix for the generated parameter vector.
    name
        Optional name for the constructed :class:`~qiskit.QuantumCircuit`.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Raises
    ------
    ValueError
        If invalid input values are provided.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if depth < 1:
        raise ValueError("depth must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    total_params = num_rot_layers * n if share_parameters else depth * num_rot_layers * n
    params = ParameterVector(parameter_prefix, total_params)

    pairs = _resolve_entanglement(n, entanglement)

    def _rot(depth_idx: int, layer_idx: int) -> None:
        base = (depth_idx * num_rot_layers + layer_idx) * n if not share_parameters else layer_idx * n
        for q in range(n):
            qc.ry(params[base + q], q)

    def _ent(depth_idx: int) -> None:
        for (i, j) in pairs:
            qc.cz(i, j)
        if star_entanglement:
            for q in range(1, n):
                qc.cz(0, q)

    for d in range(depth):
        for r in range(reps):
            _rot(d, r)
            if insert_barriers:
                qc.barrier()
            _ent(d)
            if insert_barriers:
                qc.barrier()

    if not skip_final_rotation_layer:
        _rot(depth - 1, reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZExtended(QuantumCircuit):
    """Convenience subclass for the extended RealAmplitudesCZ ansatz.

    This subclass forwards all arguments to :func:`real_amplitudes_cz_extended`
    and then composes the resulting circuit into itself.  It also exposes
    the ``input_params`` and ``num_rot_layers`` attributes for downstream
    parameter binding and analysis.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        depth: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        star_entanglement: bool = False,
        share_parameters: bool = True,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            depth,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            star_entanglement,
            share_parameters,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
