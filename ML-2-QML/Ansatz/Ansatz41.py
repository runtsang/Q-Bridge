"""RealAmplitudesCZSymmetry – a parity‑symmetric variant of the CZ‑based RealAmplitudes ansatz.

The circuit is identical to the original RealAmplitudesCZ except that an optional
parameter `enforce_symmetry` can be set to True.  When this flag is enabled all
single‑qubit rotation layers use `rz` instead of `ry`.  Because `rz` commutes
with the global parity operator `Z⊗…⊗Z`, the resulting circuit is invariant
under that symmetry.  The rest of the circuit – the entangling CZ layers and
the option to skip the final rotation layer – remains unchanged.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

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
        Total number of qubits in the circuit.
    entanglement : str or sequence or callable
        Specification of which qubit pairs should receive an entangling gate.
        Accepted string values are ``"full"``, ``"linear"``, and ``"circular"``.
        A callable is invoked with ``num_qubits`` and must return an iterable
        of ``(i, j)`` pairs.  A raw sequence of pairs can also be supplied.

    Returns
    -------
    List[Tuple[int, int]]
        A list of distinct qubit pairs ready for entanglement.

    Raises
    ------
    ValueError
        If an invalid entanglement specification is provided or a pair refers
        to an out‑of‑range qubit index.
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


def real_amplitudes_cz_symmetry(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    enforce_symmetry: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build a CZ‑based RealAmplitudes circuit that optionally respects a global Z₂ symmetry.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of entangling layers.
    entanglement : str or sequence or callable, default "full"
        Which qubit pairs receive a CZ gate.
    skip_final_rotation_layer : bool, default False
        If True, the final rotation layer is omitted.
    insert_barriers : bool, default False
        If True, insert a barrier before and after each entangling layer.
    parameter_prefix : str, default "theta"
        Prefix used for the parameter vector names.
    enforce_symmetry : bool, default False
        When True, use ``rz`` instead of ``ry`` for all rotation layers, ensuring
        invariance under the global parity operator ``Z⊗…⊗Z``.
    name : str or None, default None
        Name of the resulting quantum circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz.  The circuit exposes ``input_params`` and
        ``num_rot_layers`` attributes for introspection.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1 or if an invalid entanglement
        specification is supplied.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZSymmetry")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        base = layer * n
        if enforce_symmetry:
            # Use Rz rotations to preserve parity
            for q in range(n):
                qc.rz(params[base + q], q)
        else:
            # Default behaviour: Ry rotations
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


class RealAmplitudesCZSymmetry(QuantumCircuit):
    """Convenience wrapper class for the symmetry‑aware CZ‑based RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of entangling layers.
    entanglement : str or sequence or callable, default "full"
        Which qubit pairs receive a CZ gate.
    skip_final_rotation_layer : bool, default False
        If True, the final rotation layer is omitted.
    insert_barriers : bool, default False
        If True, insert a barrier before and after each entangling layer.
    parameter_prefix : str, default "theta"
        Prefix used for the parameter vector names.
    enforce_symmetry : bool, default False
        When True, use ``rz`` instead of ``ry`` for all rotation layers.
    name : str, default "RealAmplitudesCZSymmetry"
        Name of the quantum circuit.

    Notes
    -----
    The class simply builds the underlying circuit via :func:`real_amplitudes_cz_symmetry`
    and then composes it into a subclass of :class:`qiskit.QuantumCircuit`.  The
    attributes ``input_params`` and ``num_rot_layers`` are preserved for
    downstream use.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        enforce_symmetry: bool = False,
        name: str = "RealAmplitudesCZSymmetry",
    ) -> None:
        built = real_amplitudes_cz_symmetry(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            enforce_symmetry,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZSymmetry", "real_amplitudes_cz_symmetry"]
