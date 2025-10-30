"""
RealAmplitudesCZExtended
========================

An extended version of the ``RealAmplitudesCZ`` ansatz that adds
* depth‑controlled rotation sub‑layers,
* optional hybrid rotation styles (single RY or full RY‑RZ‑RY),
* configurable entanglement styles (CZ, XX, or both),
* and a flexible entanglement schedule.

The API mirrors the original ansatz but provides the following
new knobs:

```
depth_per_layer   # number of rotation sub‑layers per repetition
rotation_style    # "ry" or "ryrzry" (default "ry")
entanglement_style # "cz", "xx", or "cz_xx"
```

All other parameters (``reps``, ``entanglement``, ``skip_final_rotation_layer``,
``insert_barriers``, ``parameter_prefix``, ``name``) behave as in the seed
implementation.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# -----------------------------------------------------------------
# Helper: Entanglement pair resolution
# -----------------------------------------------------------------
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve a user supplied entanglement specification into a list of
    qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the entanglement pattern.

    Returns
    -------
    List[Tuple[int, int]]
        List of distinct qubit pairs.

    Raises
    ------
    ValueError
        If an unknown string is supplied or a pair is invalid.
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


# -----------------------------------------------------------------
# Main construction function
# -----------------------------------------------------------------
def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    depth_per_layer: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    entanglement_style: str = "cz",
    rotation_style: str = "ry",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct an extended RealAmplitudes ansatz with CZ/XX entanglement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    reps : int, default 1
        Number of repetition blocks.
    depth_per_layer : int, default 1
        Number of rotation sub‑layers per repetition.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement pattern to use.
    entanglement_style : str, default "cz"
        One of ``"cz"``, ``"xx"``, or ``"cz_xx"``.  Determines which
        two‑qubit gates are applied after each rotation sub‑layer.
    rotation_style : str, default "ry"
        One of ``"ry"`` (single RY per qubit) or ``"ryrzry"`` (full 3‑parameter
        RY‑RZ‑RY sequence).
    skip_final_rotation_layer : bool, default False
        If ``True`` the final rotation layer after the last repetition is omitted.
    insert_barriers : bool, default False
        If ``True`` insert a barrier after each sub‑layer and entanglement block.
    parameter_prefix : str, default "theta"
        Prefix for the parameter vector.
    name : str | None, default None
        Name of the circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz.

    Raises
    ------
    ValueError
        If an invalid configuration is supplied.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if depth_per_layer < 1:
        raise ValueError("depth_per_layer must be >= 1.")
    if rotation_style not in {"ry", "ryrzry"}:
        raise ValueError(f'rotation_style must be "ry" or "ryrzry", got {rotation_style!r}.')
    if entanglement_style not in {"cz", "xx", "cz_xx"}:
        raise ValueError(f'entanglement_style must be "cz", "xx", or "cz_xx", got {entanglement_style!r}.')

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    # Determine number of parameters per qubit per sub‑layer
    param_per_qubit = 1 if rotation_style == "ry" else 3

    # Total number of rotation sub‑layers
    total_sub_layers = reps * depth_per_layer + (0 if skip_final_rotation_layer else 1)

    # Parameter vector
    params = ParameterVector(parameter_prefix, total_sub_layers * n * param_per_qubit)

    # Resolve entanglement pairs once; they are reused
    ent_pairs = _resolve_entanglement(n, entanglement)

    # Helper to apply a rotation sub‑layer
    def _apply_rot(sub_layer: int) -> None:
        base = sub_layer * n * param_per_qubit
        for q in range(n):
            if rotation_style == "ry":
                qc.ry(params[base + q], q)
            else:  # "ryrzry"
                qc.ry(params[base + q], q)
                qc.rz(params[base + n + q], q)
                qc.ry(params[base + 2 * n + q], q)

    # Helper to apply entanglement
    def _apply_entanglement() -> None:
        if entanglement_style in {"cz", "cz_xx"}:
            for (i, j) in ent_pairs:
                qc.cz(i, j)
        if entanglement_style in {"xx", "cz_xx"}:
            for (i, j) in ent_pairs:
                qc.cx(i, j)
                qc.cx(j, i)  # equivalent to XX via two CNOTs

    # Build ansatz
    sub_layer_idx = 0
    for r in range(reps):
        for d in range(depth_per_layer):
            _apply_rot(sub_layer_idx)
            sub_layer_idx += 1
            if insert_barriers:
                qc.barrier()
            _apply_entanglement()
            if insert_barriers:
                qc.barrier()
    if not skip_final_rotation_layer:
        _apply_rot(sub_layer_idx)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = total_sub_layers  # type: ignore[attr-defined]
    return qc


# -----------------------------------------------------------------
# Class wrapper
# -----------------------------------------------------------------
class RealAmplitudesCZExtended(QuantumCircuit):
    """
    Class wrapper for the extended RealAmplitudesCZ ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    reps : int, default 1
        Number of repetition blocks.
    depth_per_layer : int, default 1
        Number of rotation sub‑layers per repetition.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement pattern.
    entanglement_style : str, default "cz"
        One of ``"cz"``, ``"xx"``, or ``"cz_xx"``.
    rotation_style : str, default "ry"
        One of ``"ry"`` or ``"ryrzry"``.
    skip_final_rotation_layer : bool, default False
    insert_barriers : bool, default False
    parameter_prefix : str, default "theta"
    name : str, default "RealAmplitudesCZExtended"
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        depth_per_layer: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        entanglement_style: str = "cz",
        rotation_style: str = "ry",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            depth_per_layer,
            entanglement,
            entanglement_style,
            rotation_style,
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
