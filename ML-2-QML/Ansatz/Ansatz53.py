"""RealAmplitudesAlternatingSymmetric ansatz.

This ansatz is a controlled modification of the original
`RealAmplitudesAlternating`.  It enforces a global qubit-permutation symmetry
by sharing rotation parameters between mirror qubits.  The alternating
RY/RX rotation pattern and entanglement schedule are preserved, but the
parameter count is reduced to roughly half of the original.

The module exposes both a convenience constructor function
`real_amplitudes_alternating_symmetric` and a subclass
`RealAmplitudesAlternatingSymmetric` of :class:`qiskit.QuantumCircuit`.

"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> list[Tuple[int, int]]:
    """Translate an entanglement specification into a list of qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Either a keyword describing a standard pattern ('full', 'linear',
        'circular') or a custom sequence/callable returning the desired
        two‑qubit pairs.

    Returns
    -------
    list[Tuple[int, int]]
        A validated list of distinct qubit pairs suitable for CX gates.

    Raises
    ------
    ValueError
        If an unknown keyword is supplied or a pair contains out‑of-range
        indices or self‑loops.
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


def real_amplitudes_alternating_symmetric(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build a symmetric real‑amplitude ansatz with alternating RY/RX layers.

    The ansatz applies a rotation to each qubit in an alternating pattern
    (RY on even layers, RX on odd layers).  To enforce global symmetry,
    qubits that are mirrors of each other around the centre share the same
    rotation parameter.  This reduces the number of free parameters to
    ``ceil(n/2)`` per layer.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit. Must be >= 1.
    reps : int, optional
        Number of entanglement‑rotation blocks. Default is 1.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], optional
        Entanglement specification.  Accepts the same options as the
        original ansatz.  Default is 'full'.
    skip_final_rotation_layer : bool, optional
        If True, the final rotation layer after the last entanglement block
        is omitted.  Default is False.
    insert_barriers : bool, optional
        When True, a barrier is inserted after each rotation and entanglement
        block for easier visualisation.  Default is False.
    parameter_prefix : str, optional
        Prefix used for generated parameters.  Default is 'theta'.
    name : str | None, optional
        Circuit name.  If None, a default name is chosen.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit ready for binding and execution.

    Notes
    -----
    - The total number of parameters is ``num_rot_layers * ceil(num_qubits/2)``.
    - The symmetry is enforced by mirroring indices: for qubit *q* the
      parameter index is ``min(q, n-1-q)``.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingSymmetric")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    half_params_per_layer = (n + 1) // 2  # ceil(n/2)
    params = ParameterVector(parameter_prefix, num_rot_layers * half_params_per_layer)

    def _rot(layer: int) -> None:
        """Apply a rotation layer with symmetry."""
        base = layer * half_params_per_layer
        for q in range(n):
            mirror = n - 1 - q
            idx = min(q, mirror)
            param = params[base + idx]
            if layer % 2 == 0:
                qc.ry(param, q)
            else:
                qc.rx(param, q)

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


class RealAmplitudesAlternatingSymmetric(QuantumCircuit):
    """Convenience wrapper for the symmetric alternating‑rotation ansatz.

    The class behaves like a normal :class:`qiskit.QuantumCircuit` and
    exposes the same configuration knobs as :func:`real_amplitudes_alternating_symmetric`.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingSymmetric",
    ) -> None:
        built = real_amplitudes_alternating_symmetric(
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


__all__ = ["RealAmplitudesAlternatingSymmetric", "real_amplitudes_alternating_symmetric"]
