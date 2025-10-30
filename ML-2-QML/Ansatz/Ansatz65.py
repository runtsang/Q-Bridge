"""Extended RealAmplitudes ansatz.

This module adds two optional layers to the standard
RealAmplitudes construction:
  * **Mixing layer** – a parallel RZ rotation on each qubit.
  * **Additional entanglement** – a parameterised RZZ gate applied
    after each CX entanglement block.

The ansatz remains fully parameterised and compatible with
Qiskit's `QuantumCircuit` API.

Usage
-----
>>> from ansatz_scaled.real_amplitudes_extension import real_amplitudes_extended
>>> qc = real_amplitudes_extended(num_qubits=4, reps=2,
...                               mixing_layer=True,
...                               additional_entanglement=True)
>>> print(qc)
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter


def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ],
    entanglement_depth: float | None = None,
) -> List[Tuple[int, int]]:
    """Return a deterministic list of two‑qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence | Callable
        Entanglement specification.  Valid strings are
        ``"full"``, ``"linear"``, and ``"circular"``.  A custom
        sequence or callable may be supplied.
    entanglement_depth : float | None
        If provided and ``0 < depth < 1`` only the first
        ``floor(len(pairs) * depth)`` pairs are kept.  This
        offers a lightweight way to reduce the entanglement
        density.

    Returns
    -------
    List[Tuple[int, int]]
        A list of distinct qubit pairs.

    Raises
    ------
    ValueError
        If the specification is invalid or out of range.
    """
    if isinstance(entanglement, str):
        if entanglement == "full":
            pairs = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        elif entanglement == "linear":
            pairs = [(i, i + 1) for i in range(num_qubits - 1)]
        elif entanglement == "circular":
            pairs = [(i, i + 1) for i in range(num_qubits - 1)]
            if num_qubits > 2:
                pairs.append((num_qubits - 1, 0))
        else:
            raise ValueError(f"Unknown entanglement string: {entanglement!r}")
    elif callable(entanglement):
        pairs = list(entanglement(num_qubits))
    else:
        pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]

    # Validate pairs
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")

    # Optional depth reduction
    if entanglement_depth is not None:
        if not (0 < entanglement_depth <= 1):
            raise ValueError("entanglement_depth must be in (0, 1].")
        keep = max(1, int(len(pairs) * entanglement_depth))
        pairs = pairs[:keep]

    return pairs


def real_amplitudes_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ] = "full",
    entanglement_depth: float | None = None,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    mixing_layer: bool = False,
    additional_entanglement: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended RealAmplitudes-style circuit.

    The extended ansatz consists of the following repeating block:

    1. **Rotation layer** – RY gates on all qubits.
    2. **Mixing layer** – optional RZ gates on all qubits.
    3. **Entanglement** – CX gates according to *entanglement*.
    4. **Additional entanglement** – optional RZZ gates on each entanglement
       pair.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of repeating blocks.
    entanglement : str | Sequence | Callable, default "full"
        Entanglement pattern as described in :func:`_resolve_entanglement`.
    entanglement_depth : float | None, default None
        Fraction of entanglement pairs to keep per layer.
    skip_final_rotation_layer : bool, default False
        If ``True`` the final RY layer after the last block is omitted.
    insert_barriers : bool, default False
        Insert a barrier after each block.
    mixing_layer : bool, default False
        If ``True`` add an RZ rotation on each qubit after the RY layer.
    additional_entanglement : bool, default False
        If ``True`` add an RZZ gate on each entanglement pair after the CX
        block.  Each RZZ receives its own parameter.
    parameter_prefix : str, default "theta"
        Prefix for all parameters.
    name : str | None, default None
        Circuit name.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit with the extended RealAmplitudes pattern.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1 or inputs are invalid.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesExtended")

    # Resolve entanglement pairs once; they are reused each layer.
    pairs = _resolve_entanglement(n, entanglement, entanglement_depth)
    num_pairs = len(pairs)

    # Parameter bookkeeping
    rot_per_rep = n
    mix_per_rep = n if mixing_layer else 0
    ent_per_rep = num_pairs if additional_entanglement else 0

    total_params = rot_per_rep * reps + mix_per_rep * reps + ent_per_rep * reps
    if not skip_final_rotation_layer:
        total_params += n  # final rotation layer

    params = ParameterVector(parameter_prefix, total_params)

    # Offsets for parameter slices
    rot_offset = 0
    mix_offset = rot_offset + rot_per_rep * reps
    ent_offset = mix_offset + mix_per_rep * reps
    final_rot_offset = None
    if not skip_final_rotation_layer:
        final_rot_offset = ent_offset + ent_per_rep * reps

    def _rotation_layer(layer_idx: int, offset: int) -> None:
        base = offset + layer_idx * n
        for q in range(n):
            qc.ry(params[base + q], q)

    def _mixing_layer_func(layer_idx: int, offset: int) -> None:
        if not mixing_layer:
            return
        base = offset + layer_idx * n
        for q in range(n):
            qc.rz(params[base + q], q)

    def _additional_entanglement(layer_idx: int, offset: int) -> None:
        if not additional_entanglement:
            return
        base = offset + layer_idx * num_pairs
        for idx, (i, j) in enumerate(pairs):
            qc.rzz(params[base + idx], i, j)

    # Build the circuit
    for r in range(reps):
        _rotation_layer(r, rot_offset)
        _mixing_layer_func(r, mix_offset)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()
        _additional_entanglement(r, ent_offset)

    if not skip_final_rotation_layer:
        _rotation_layer(reps, final_rot_offset)

    # Attach metadata for introspection
    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = reps + (0 if skip_final_rotation_layer else 1)  # type: ignore[attr-defined]
    qc.num_mixing_layers = reps if mixing_layer else 0  # type: ignore[attr-defined]
    qc.num_entanglement_layers = reps if additional_entanglement else 0  # type: ignore[attr-defined]
    qc.entanglement_pairs = pairs  # type: ignore[attr-defined]
    return qc


class RealAmplitudesExtended(QuantumCircuit):
    """Convenience wrapper around :func:`real_amplitudes_extended`.

    The constructor builds the circuit using the same parameter
    names and configuration knobs as the free function.  It
    behaves like any other Qiskit `QuantumCircuit` instance
    (supports `compose`, `bind_parameters`, etc.).

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of repeating blocks.
    entanglement : str | Sequence | Callable, default "full"
        Entanglement pattern.
    entanglement_depth : float | None, default None
        Fraction of entanglement pairs to keep per layer.
    skip_final_rotation_layer : bool, default False
        If ``True`` the final RY layer after the last block is omitted.
    insert_barriers : bool, default False
        Insert a barrier after each block.
    mixing_layer : bool, default False
        If ``True`` add an RZ rotation on each qubit after the RY layer.
    additional_entanglement : bool, default False
        If ``True`` add an RZZ gate on each entanglement pair after the CX block.
    parameter_prefix : str, default "theta"
        Prefix for all parameters.
    name : str, default "RealAmplitudesExtended"
        Circuit name.
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
        entanglement_depth: float | None = None,
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        mixing_layer: bool = False,
        additional_entanglement: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesExtended",
    ) -> None:
        built = real_amplitudes_extended(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            entanglement_depth=entanglement_depth,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            mixing_layer=mixing_layer,
            additional_entanglement=additional_entanglement,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        # Preserve metadata
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.num_mixing_layers = built.num_mixing_layers  # type: ignore[attr-defined]
        self.num_entanglement_layers = built.num_entanglement_layers  # type: ignore[attr-defined]
        self.entanglement_pairs = built.entanglement_pairs  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesExtended", "real_amplitudes_extended"]
