"""
RealAmplitudesAlternatingControlled ansatz with optional symmetry constraint.

* Adds a `symmetry` flag that, when enabled, forces all qubits in a given rotation
  layer to share a single rotation parameter.  This reduces the number of free
  parameters and imposes a global rotational symmetry.
* Keeps the alternating RY/RX rotation pattern and the entanglement schedule
  identical to the seed implementation.
* Provides a convenience function `real_amplitudes_alternating_controlled` and
  a subclass `RealAmplitudesAlternatingControlled` that extends `QuantumCircuit`.
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
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification. Accepted string values are
        ``'full'`` (all‑to‑all), ``'linear'`` (nearest‑neighbour chain) and
        ``'circular'`` (chain with a wrap‑around CNOT).  Alternatively a
        sequence of explicit pairs or a callable that returns such a sequence
        may be supplied.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of distinct qubit pairs.

    Raises
    ------
    ValueError
        If the specification is invalid or contains out‑of‑range indices.
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
    parameter_prefix: str = "theta",
    symmetry: bool = True,
    name: str | None = None,
) -> QuantumCircuit:
    """Build a controlled‑modification RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, optional
        Number of alternating rotation–entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], optional
        Entanglement pattern. Accepted values are ``'full'``, ``'linear'`` and
        ``'circular'`` or a custom sequence/callable.
    skip_final_rotation_layer : bool, optional
        If True, the final rotation layer after the last entanglement block is omitted.
    insert_barriers : bool, optional
        Insert a barrier before/after each entanglement block for clearer circuit
        visualization.
    parameter_prefix : str, optional
        Prefix for the generated rotation parameters.
    symmetry : bool, optional
        When True, all qubits in a rotation layer share a single parameter.  When
        False, the original per‑qubit parameters are used.
    name : str, optional
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1 or if the entanglement specification
        is invalid.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingControlled")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n if not symmetry else num_rot_layers)

    def _rot(layer: int) -> None:
        """Apply a single rotation layer."""
        if symmetry:
            param = params[layer]
            if layer % 2 == 0:
                for q in range(n):
                    qc.ry(param, q)
            else:
                for q in range(n):
                    qc.rx(param, q)
        else:
            base = layer * n
            if layer % 2 == 0:
                for q in range(n):
                    qc.ry(params[base + q], q)
            else:
                for q in range(n):
                    qc.rx(params[base + q], q)

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


class RealAmplitudesAlternatingControlled(QuantumCircuit):
    """Convenience subclass for the controlled‑symmetry RealAmplitudes ansatz.

    The subclass exposes the same configuration knobs as the function
    ``real_amplitudes_alternating_controlled`` while allowing direct
    composition as a ``QuantumCircuit`` instance.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, optional
        Number of alternating rotation–entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], optional
        Entanglement pattern.
    skip_final_rotation_layer : bool, optional
        Skip the final rotation layer.
    insert_barriers : bool, optional
        Insert barriers between layers.
    parameter_prefix : str, optional
        Parameter name prefix.
    symmetry : bool, optional
        Symmetry flag controlling parameter sharing.
    name : str, optional
        Circuit name.
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
        name: str = "RealAmplitudesAlternatingControlled",
    ) -> None:
        built = real_amplitudes_alternating_controlled(
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


__all__ = ["RealAmplitudesAlternatingControlled", "real_amplitudes_alternating_controlled"]
