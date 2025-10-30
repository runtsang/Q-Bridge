"""Extended RealAmplitudes ansatz with richer expressivity.

Features
--------
- **Entanglement schedule**: allow a different entanglement pattern per repetition.
- **Hybrid rotations**: choose between pure RY or RZ+RY layers.
- **Parameter sharing**: reuse the same parameters across all rotation layers.
- **Additional entanglement patterns**: ``full``, ``linear``, ``circular``, ``star`` (center qubit entangled with all others), ``random``.
- **Barrier insertion** and **final rotation layer** control.

The module exposes a convenience constructor ``extended_real_amplitudes`` and a subclass
``ExtendedRealAmplitudes`` that behaves like a Qiskit ``QuantumCircuit``.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Union

import random

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
    *,
    rng: random.Random | None = None,
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to an entanglement spec.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Specification of the entanglement pattern.
    rng : random.Random | None
        Optional random generator for reproducible ``random`` patterns.

    Supported string values
    -----------------------
    - ``full``   : all possible pairs.
    - ``linear`` : ``(i, i+1)`` for all qubits.
    - ``circular`` : linear plus an edge between the last and first qubit.
    - ``star``   : qubit 0 entangled with all others.
    - ``random`` : a random set of disjoint pairs covering each qubit at most once.

    Raises
    ------
    ValueError
        If an unknown string is supplied or if any pair is invalid.
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
        if entanglement == "star":
            if num_qubits < 2:
                raise ValueError("Star entanglement requires at least 2 qubits.")
            return [(0, i) for i in range(1, num_qubits)]
        if entanglement == "random":
            if num_qubits < 2:
                raise ValueError("Random entanglement requires at least 2 qubits.")
            rng = rng or random
            # Shuffle qubits and pair them sequentially
            qubits = list(range(num_qubits))
            rng.shuffle(qubits)
            return [(qubits[i], qubits[i + 1]) for i in range(0, num_qubits - 1, 2)]
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


def extended_real_amplitudes(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    entanglement_schedule: Sequence[
        Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]]
    ] | None = None,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    rotation_type: str = "ry",
    share_params: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended RealAmplitudes-style ``QuantumCircuit``.

    Parameters
    ----------
    num_qubits : int
        Number of qubits to construct.
    reps : int
        Number of rotation‑entanglement repetitions.
    entanglement : str | Sequence | Callable
        Default entanglement pattern for all repetitions if
        ``entanglement_schedule`` is ``None``.
    entanglement_schedule : Sequence | None
        Optional per‑repetition entanglement specification. Length must equal
        ``reps``. Each element can be a string, a list of pairs, or a callable.
    skip_final_rotation_layer : bool
        If ``True`` the last rotation layer is omitted.
    insert_barriers : bool
        Insert a barrier after each rotation and entanglement block.
    rotation_type : str
        ``"ry"`` for RY rotations only, ``"rz_ry"`` for RZ followed by RY.
    share_params : bool
        If ``True`` all rotation layers reuse the same set of parameters.
    parameter_prefix : str
        Prefix used for parameter names.
    name : str | None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "ExtendedRealAmplitudes")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    total_params = n if share_params else n * num_rot_layers
    params = ParameterVector(parameter_prefix, total_params)

    # Helper to apply a rotation layer
    def _rotation_layer(layer_idx: int) -> None:
        base = 0 if share_params else layer_idx * n
        for q in range(n):
            param = params[base + q]
            if rotation_type == "ry":
                qc.ry(param, q)
            elif rotation_type == "rz_ry":
                qc.rz(param, q)
                qc.ry(param, q)
            else:
                raise ValueError(f"Unsupported rotation_type: {rotation_type!r}")

    # Resolve entanglement pairs for each repetition
    if entanglement_schedule is not None:
        if len(entanglement_schedule)!= reps:
            raise ValueError(
                "entanglement_schedule must have the same length as reps."
            )
        schedule = entanglement_schedule
    else:
        schedule = [entanglement] * reps

    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        pairs = _resolve_entanglement(n, schedule[r])
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    qc.rotation_type = rotation_type  # type: ignore[attr-defined]
    qc.share_params = share_params  # type: ignore[attr-defined]
    qc.entanglement_schedule = entanglement_schedule  # type: ignore[attr-defined]
    return qc


class ExtendedRealAmplitudes(QuantumCircuit):
    """Class‐style wrapper for the extended RealAmplitudes ansatz.

    The constructor forwards all arguments to :func:`extended_real_amplitudes`
    and then composes the resulting circuit into the instance.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        entanglement_schedule: Sequence[
            Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]]
        ] | None = None,
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        rotation_type: str = "ry",
        share_params: bool = False,
        parameter_prefix: str = "theta",
        name: str = "ExtendedRealAmplitudes",
    ) -> None:
        built = extended_real_amplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            entanglement_schedule=entanglement_schedule,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            rotation_type=rotation_type,
            share_params=share_params,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.rotation_type = built.rotation_type  # type: ignore[attr-defined]
        self.share_params = built.share_params  # type: ignore[attr-defined]
        self.entanglement_schedule = built.entanglement_schedule  # type: ignore[attr-defined]


__all__ = ["ExtendedRealAmplitudes", "extended_real_amplitudes"]
