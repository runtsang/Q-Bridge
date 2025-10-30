"""RealAmplitudes variant with alternating RY/RX rotation layers and optional symmetry constraints.

This module implements a controlled modification of the standard alternating Real Amplitudes ansatz.
Two key features are added:

* **Mirror symmetry** – Rotation parameters for qubits that are symmetric about the centre of the register
  are shared.  For a register of length *n*, qubits *i* and *n‑1‑i* use the same parameter.
* **Mirror entanglement** – When requested, each qubit is entangled with its mirror partner instead of
  using the default entanglement schedule.

The ansatz remains compatible with :class:`qiskit.circuit.QuantumCircuit` and can be composed,
parameterised and executed on any Qiskit backend.

Both a convenience constructor function and a subclass of :class:`~qiskit.circuit.QuantumCircuit`
are provided.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to an entanglement specification.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the register.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Description of the entanglement pattern.  The following strings are recognised:

        * ``"full"`` – all pairs of qubits are entangled.
        * ``"linear"`` – nearest‑neighbour couplings.
        * ``"circular"`` – linear couplings plus a link between the last and the first qubit.
        * ``"mirror"`` – each qubit is coupled to its mirror partner (qubit *i* with *n‑1‑i*).
        * A custom list of tuples or a callable that returns such a list.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs for two‑qubit gates.

    Raises
    ------
    ValueError
        If an unknown string is supplied or if a pair contains identical qubits or
        qubits outside the register.
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


def real_amplitudes_alternating_symmetric(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    mirror_entanglement: bool = False,
    symmetry: bool = True,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build a symmetry‑aware alternating Real Amplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit. Must be positive.
    reps : int, default 1
        Number of alternating rotation‑entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement pattern.  ``"mirror"`` is supported when ``mirror_entanglement`` is
        ``True``; otherwise the string is interpreted as in :func:`_resolve_entanglement`.
    skip_final_rotation_layer : bool, default False
        If ``True``, the final rotation layer after the last repetition is omitted.
    insert_barriers : bool, default False
        Insert a barrier after each rotation or entanglement block for visual clarity.
    mirror_entanglement : bool, default False
        When ``True``, the entanglement pattern is forced to the mirror pairing,
        ignoring the ``entanglement`` argument except for type checking.
    symmetry : bool, default True
        When ``True``, rotation parameters are shared between symmetric qubits.
    parameter_prefix : str, default "theta"
        Prefix used to name the parameters in the :class:`~qiskit.circuit.ParameterVector`.
    name : str | None, default None
        Name of the resulting :class:`~qiskit.circuit.QuantumCircuit`.  If ``None``,
        ``"RealAmplitudesAlternatingSymmetric"`` is used.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit implementing the symmetric alternating Real Amplitudes
        ansatz.

    Raises
    ------
    ValueError
        If ``num_qubits`` is not positive or if an invalid entanglement specification
        is supplied.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingSymmetric")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    # number of parameters per layer depends on symmetry flag
    params_per_layer = n if not symmetry else (n + 1) // 2
    params = ParameterVector(parameter_prefix, num_rot_layers * params_per_layer)

    def _rot(layer: int) -> None:
        base = layer * params_per_layer
        if layer % 2 == 0:
            for q in range(n):
                idx = base + (q if not symmetry else (q if q < n // 2 else n - 1 - q))
                qc.ry(params[idx], q)
        else:
            for q in range(n):
                idx = base + (q if not symmetry else (q if q < n // 2 else n - 1 - q))
                qc.rx(params[idx], q)

    # Resolve entanglement pairs, overriding if mirror entanglement requested
    if mirror_entanglement:
        pairs = _resolve_entanglement(n, "mirror")
    else:
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
    """Convenience wrapper for the symmetry‑aware alternating Real Amplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of alternating rotation‑entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement pattern.  ``"mirror"`` is supported when ``mirror_entanglement`` is
        ``True``; otherwise the string is interpreted as in :func:`_resolve_entanglement`.
    skip_final_rotation_layer : bool, default False
        If ``True``, the final rotation layer after the last repetition is omitted.
    insert_barriers : bool, default False
        Insert a barrier after each rotation or entanglement block for visual clarity.
    mirror_entanglement : bool, default False
        When ``True``, the entanglement pattern is forced to the mirror pairing,
        ignoring the ``entanglement`` argument except for type checking.
    symmetry : bool, default True
        When ``True``, rotation parameters are shared between symmetric qubits.
    parameter_prefix : str, default "theta"
        Prefix used to name the parameters in the :class:`~qiskit.circuit.ParameterVector`.
    name : str, default "RealAmplitudesAlternatingSymmetric"
        Name of the circuit.

    Notes
    -----
    The constructor simply builds the underlying circuit via
    :func:`real_amplitudes_alternating_symmetric` and composes it into ``self``.
    """
    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        mirror_entanglement: bool = False,
        symmetry: bool = True,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingSymmetric",
    ) -> None:
        built = real_amplitudes_alternating_symmetric(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            mirror_entanglement,
            symmetry,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = [
    "RealAmplitudesAlternatingSymmetric",
    "real_amplitudes_alternating_symmetric",
]
