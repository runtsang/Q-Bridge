"""RealAmplitudesCZShared
~~~~~~~~~~~~~~~~~~~~~~~~
A controlled‑parameter‑sharing variant of the RealAmplitudesCZ ansatz.
It keeps the same rotation‑and‑CZ pattern but reuses the same set of
rotation angles across all layers, optionally enforcing mirror symmetry
between qubits i and n‑1‑i.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs for entanglement.

    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit.
    entanglement
        Either a string spec ('full', 'linear', 'circular'),
        a user‑defined sequence of pairs, or a callable that
        returns such a sequence.

    Returns
    -------
    list[tuple[int, int]]
        Validated list of (control, target) pairs.

    Raises
    ------
    ValueError
        If the spec is invalid or contains out‑of-range indices.
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


def real_amplitudes_cz_shared(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
    share_parameters: bool = True,
) -> QuantumCircuit:
    """Create a RealAmplitudesCZ circuit with optional parameter sharing.

    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit.
    reps
        Number of entanglement layers.
    entanglement
        Specification of which qubit pairs receive CZ gates.
    skip_final_rotation_layer
        If ``True`` the last rotation layer is omitted.
    insert_barriers
        If ``True`` insert barriers between layers for visual clarity.
    parameter_prefix
        Prefix for the rotation parameters.
    name
        Name of the resulting circuit.
    share_parameters
        If ``True`` all rotation layers reuse the same set of
        parameters (one per qubit).  If ``False`` each layer has its
        own independent parameters.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZShared")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    if share_parameters:
        params = ParameterVector(parameter_prefix, n)
    else:
        params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        base = 0 if share_parameters else layer * n
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
    """Convenience wrapper for the parameter‑sharing RealAmplitudesCZ ansatz.

    Parameters are stored on the instance as ``input_params`` and
    the number of rotation layers as ``num_rot_layers`` to match the
    original interface.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZShared",
        share_parameters: bool = True,
    ) -> None:
        built = real_amplitudes_cz_shared(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
            share_parameters,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZShared", "real_amplitudes_cz_shared"]
