"""Enhanced RealAmplitudes ansatz with deeper expressivity and configurable entanglement.

This module extends the canonical RealAmplitudes ansatz by providing:
* additional entanglement patterns (star, random),
* configurable entanglement gate (CX or CZ),
* rotation gate choice (RY, RX, or hybrid),
* entanglement depth control,
* optional parameter sharing across layers,
* optional barrier placement.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Optional
import random

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement_extended(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
    seed: int | None = None,
) -> List[Tuple[int, int]]:
    """
    Translate an entanglement specification into a list of two‑qubit pairs.

    Supported string patterns:
        * ``"full"``     – every distinct pair
        * ``"linear"``   – nearest‑neighbour chain
        * ``"circular"`` – linear chain with a closing edge
        * ``"star"``     – all qubits entangled with qubit 0
        * ``"random"``   – random unique pairs, optionally seeded

    For custom specifications the function accepts a sequence of tuples or a
    callable that returns such a sequence.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement definition.
    seed : int | None
        Optional seed for reproducible random entanglement.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs.

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
        if entanglement == "star":
            return [(0, i) for i in range(1, num_qubits)] if num_qubits > 1 else []
        if entanglement == "random":
            rng = random.Random(seed)
            pairs: set[Tuple[int, int]] = set()
            while len(pairs) < max(1, num_qubits):
                i, j = rng.sample(range(num_qubits), 2)
                pairs.add(tuple(sorted((i, j))))
            return list(pairs)
        raise ValueError(f"Unknown entanglement string: {entanglement!r}")

    if callable(entanglement):
        pairs = list(entanglement(num_qubits))
        return [(int(i), int(j)) for (i, j) in pairs]

    # Assume sequence of tuples
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def real_amplitudes_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
    entanglement_gate: str = "cx",
    rotation_gate: str = "ry",
    entanglement_depth: int = 1,
    parameter_sharing: bool = False,
    layer_barriers: bool = False,
    seed: int | None = None,
) -> QuantumCircuit:
    """
    Build an extended RealAmplitudes ansatz with configurable entanglement and rotation layers.

    The ansatz consists of alternating rotation layers (RY, RX or a hybrid) and
    two‑qubit entangling gates (CX or CZ).  The number of repetitions, the
    depth of entanglement within each repetition, and whether parameters are
    shared across layers are fully user‑configurable.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of rotation–entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement schedule.  See :func:`_resolve_entanglement_extended` for
        supported string patterns.
    skip_final_rotation_layer : bool, default False
        If ``True``, the last rotation layer is omitted (useful for layer‑by‑layer
        optimization schemes).
    insert_barriers : bool, default False
        Insert a barrier after each rotation and entanglement block.
    parameter_prefix : str, default "theta"
        Prefix for the rotation parameters.
    name : str | None, default None
        Optional circuit name; defaults to ``"RealAmplitudesExtended"``.
    entanglement_gate : str, default "cx"
        Two‑qubit gate to use for entanglement.  Supported values are ``"cx"`` and
        ``"cz"``.
    rotation_gate : str, default "ry"
        Rotation gate used in the rotation layers.  Supported values are
        ``"ry"``, ``"rx"``, and ``"hybrid"`` (alternating RY and RX).
    entanglement_depth : int, default 1
        Number of times the entanglement pattern is applied within each repetition.
    parameter_sharing : bool, default False
        If ``True``, all rotation layers share the same parameter vector,
        reducing the total number of variational parameters.
    layer_barriers : bool, default False
        Insert a barrier after each full rotation–entanglement block.
    seed : int | None, default None
        Seed for reproducible random entanglement when ``entanglement="random"``.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Raises
    ------
    ValueError
        On invalid input values (e.g., negative qubits, unsupported gates).
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")
    if entanglement_depth < 1:
        raise ValueError("entanglement_depth must be >= 1.")
    if entanglement_gate.lower() not in {"cx", "cz"}:
        raise ValueError(f"Unsupported entanglement_gate: {entanglement_gate!r}")
    if rotation_gate.lower() not in {"ry", "rx", "hybrid"}:
        raise ValueError(f"Unsupported rotation_gate: {rotation_gate!r}")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesExtended")

    # Determine number of rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter vector(s)
    if parameter_sharing:
        params = ParameterVector(parameter_prefix, n)
    else:
        params = ParameterVector(parameter_prefix, num_rot_layers * n)

    # Resolve entanglement pairs
    pairs = _resolve_entanglement_extended(n, entanglement, seed=seed)

    # Helper to apply a rotation layer
    def _apply_rotation_layer(layer_idx: int) -> None:
        base = 0 if parameter_sharing else layer_idx * n
        for q in range(n):
            if rotation_gate.lower() == "ry":
                qc.ry(params[base + q], q)
            elif rotation_gate.lower() == "rx":
                qc.rx(params[base + q], q)
            else:  # hybrid
                if q % 2 == 0:
                    qc.ry(params[base + q], q)
                else:
                    qc.rx(params[base + q], q)

    # Build circuit
    for r in range(reps):
        _apply_rotation_layer(r)
        if insert_barriers or layer_barriers:
            qc.barrier()
        for _ in range(entanglement_depth):
            for (i, j) in pairs:
                if entanglement_gate.lower() == "cx":
                    qc.cx(i, j)
                else:  # cz
                    qc.cz(i, j)
        if insert_barriers or layer_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _apply_rotation_layer(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesExtended(QuantumCircuit):
    """Convenient class wrapper for the extended RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of rotation–entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement schedule.  See :func:`real_amplitudes_extended` for details.
    skip_final_rotation_layer : bool, default False
        Omit the last rotation layer.
    insert_barriers : bool, default False
        Insert barriers after each block.
    parameter_prefix : str, default "theta"
        Prefix for rotation parameters.
    name : str, default "RealAmplitudesExtended"
        Circuit name.
    entanglement_gate : str, default "cx"
        Gate used for entanglement (``"cx"`` or ``"cz"``).
    rotation_gate : str, default "ry"
        Rotation gate used in the rotation layers
        (``"ry"``, ``"rx"``, or ``"hybrid"``).
    entanglement_depth : int, default 1
        Depth of entanglement within each repetition.
    parameter_sharing : bool, default False
        Share parameters across rotation layers.
    layer_barriers : bool, default False
        Insert barriers after each full block.
    seed : int | None, default None
        Seed for reproducible random entanglement.

    Notes
    -----
    The class behaves like a standard :class:`qiskit.QuantumCircuit` and can be
    composed, parameter‑bound, or executed on a simulator or backend.
    """
    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesExtended",
        entanglement_gate: str = "cx",
        rotation_gate: str = "ry",
        entanglement_depth: int = 1,
        parameter_sharing: bool = False,
        layer_barriers: bool = False,
        seed: int | None = None,
    ) -> None:
        built = real_amplitudes_extended(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            name=name,
            entanglement_gate=entanglement_gate,
            rotation_gate=rotation_gate,
            entanglement_depth=entanglement_depth,
            parameter_sharing=parameter_sharing,
            layer_barriers=layer_barriers,
            seed=seed,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        # expose the parameters for external binding
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesExtended", "real_amplitudes_extended"]
