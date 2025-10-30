"""RealAmplitudesCZExtended: a depth‑controlled hybrid ansatz that preserves the original RealAmplitudesCZ structure while adding a configurable entanglement depth and optional entanglement gate type.

The module defines:
* A convenience function `real_amplitudes_cz_extended` that re‑implements the original circuit with added depth control and gate flexibility.
* A subclass `RealAmplitudesCZExtended` that wraps the same functionality and exposes the circuit as a `QuantumCircuit` subclass.

Design choices:
- *Depth‑scheduling*: The `entanglement_depth` parameter allows the user to repeat the entanglement layer multiple times per repetition, increasing expressivity.
- *Gate flexibility*: The `entanglement_gate` parameter lets the user choose between CZ (default) and CX entangling gates.
- *Parameter handling*: Rotation parameters are allocated via a `ParameterVector`; entanglement gates remain parameter‑free.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the entanglement pattern.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs.

    Raises
    ------
    ValueError
        If the specification is invalid or contains out‑of-range indices.
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
            raise ValueError(
                f"Entanglement pair {(i, j)} out of range for n={num_qubits}."
            )
    return pairs


def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ] = "full",
    entanglement_gate: str = "cz",
    entanglement_depth: int = 1,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Construct a RealAmplitudesCZ‑style ansatz with configurable entanglement depth.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of rotation layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern specification.
    entanglement_gate : str, default "cz"
        Gate used for entanglement; either "cz" or "cx".
    entanglement_depth : int, default 1
        Number of times the entanglement layer is repeated per repetition.
    skip_final_rotation_layer : bool, default False
        If True, omit the rotation layer after the last entanglement block.
    insert_barriers : bool, default False
        Insert a barrier after each rotation and entanglement block.
    parameter_prefix : str, default "theta"
        Prefix for the rotation parameters.
    name : str | None, default None
        Optional name for the resulting circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Raises
    ------
    ValueError
        If input arguments are invalid.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")
    if entanglement_depth < 1:
        raise ValueError("entanglement_depth must be >= 1.")
    if entanglement_gate not in {"cz", "cx"}:
        raise ValueError("entanglement_gate must be either 'cz' or 'cx'.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    # Determine rotation layer count
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        base = layer * n
        for q in range(n):
            qc.ry(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for _ in range(entanglement_depth):
            for (i, j) in pairs:
                if entanglement_gate == "cz":
                    qc.cz(i, j)
                else:  # "cx"
                    qc.cx(i, j)
            if insert_barriers:
                qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZExtended(QuantumCircuit):
    """Convenience subclass that builds a RealAmplitudesCZ‑style ansatz with configurable entanglement depth.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of rotation layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern specification.
    entanglement_gate : str, default "cz"
        Gate used for entanglement; either "cz" or "cx".
    entanglement_depth : int, default 1
        Number of times the entanglement layer is repeated per repetition.
    skip_final_rotation_layer : bool, default False
        If True, omit the rotation layer after the last entanglement block.
    insert_barriers : bool, default False
        Insert a barrier after each rotation and entanglement block.
    parameter_prefix : str, default "theta"
        Prefix for the rotation parameters.
    name : str, default "RealAmplitudesCZExtended"
        Name of the resulting circuit.

    Notes
    -----
    The instance attributes `input_params` and `num_rot_layers` are exposed for
    compatibility with parameter binding workflows.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: Union[
            str,
            Sequence[Tuple[int, int]],
            Callable[[int], Sequence[Tuple[int, int]]],
        ] = "full",
        entanglement_gate: str = "cz",
        entanglement_depth: int = 1,
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            entanglement,
            entanglement_gate,
            entanglement_depth,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]
