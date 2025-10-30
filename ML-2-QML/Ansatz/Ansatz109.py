"""RealAmplitudesExtended: a depth‑controlled, hybrid‑rotation ansatz."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = ["RealAmplitudesExtended", "real_amplitudes_extended"]


def _resolve_entanglement(
    n: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec."""
    if isinstance(entanglement, str):
        if entanglement == "full":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        if entanglement == "linear":
            return [(i, i + 1) for i in range(n - 1)]
        if entanglement == "circular":
            pairs = [(i, i + 1) for i in range(n - 1)]
            if n > 2:
                pairs.append((n - 1, 0))
            return pairs
        raise ValueError(f"Unknown entanglement string: {entanglement!r}")

    if callable(entanglement):
        pairs = list(entanglement(n))
        return [(int(i), int(j)) for (i, j) in pairs]

    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={n}.")
    return pairs


def real_amplitudes_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    *,
    hybrid: bool = False,
    entangler_schedule: Optional[Callable[[int, int], Sequence[Tuple[int, int]]]] = None,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a depth‑controlled, hybrid‑rotation ``QuantumCircuit``.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default=1
        Number of repetition cycles.
    entanglement : str or sequence or callable, default="full"
        Specification of the entangler pairs.  If a callable is provided,
        it receives ``(num_qubits, repetition)`` and returns a sequence of
        two‑qubit pairs for that repetition.
    skip_final_rotation_layer : bool, default=False
        If ``True``, the final rotation layer (RY only) is omitted.
    insert_barriers : bool, default=False
        Insert barriers between logical blocks for easier visualisation.
    parameter_prefix : str, default="theta"
        Prefix used for the ``ParameterVector`` names.
    hybrid : bool, default=False
        If ``True``, an additional RZ‑only rotation block follows each
        RY block within a repetition.
    entangler_schedule : callable, optional
        Custom schedule generator.  If ``None``, the ``entanglement``
        argument is used for every repetition.
    name : str, optional
        Optional name for the returned ``QuantumCircuit``.
    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Notes
    -----
    * The total number of parameters is
      ``(reps * (1 + hybrid) + (0 if skip_final_rotation_layer else 1)) * num_qubits``.
    * The circuit is fully compatible with Qiskit's variational
      optimisation framework.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if not isinstance(hybrid, bool):
        raise TypeError("hybrid must be a boolean.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesExtended")

    # Determine parameter vector size
    num_rot_layers = reps * (1 + int(hybrid)) + (0 if skip_final_rotation_layer else 1)
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rotation_layer(layer_idx: int, gate: str) -> None:
        base = layer_idx * n
        for q in range(n):
            if gate == "ry":
                qc.ry(params[base + q], q)
            elif gate == "rz":
                qc.rz(params[base + q], q)
            else:
                raise ValueError(f"Unsupported rotation gate: {gate!r}")

    # Resolve default entanglement pairs once for use when no custom schedule is supplied
    default_pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        # RY layer
        _rotation_layer(r * (1 + int(hybrid)), "ry")
        if insert_barriers:
            qc.barrier()

        # Entanglers for this repetition
        if entangler_schedule is None:
            pairs = default_pairs
        else:
            pairs = list(entangler_schedule(n, r))
            pairs = [(int(i), int(j)) for (i, j) in pairs]
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

        # Optional RZ layer
        if hybrid:
            _rotation_layer(r * (1 + int(hybrid)) + 1, "rz")
            if insert_barriers:
                qc.barrier()
            # Optional second entanglement after RZ
            for (i, j) in pairs:
                qc.cx(i, j)
            if insert_barriers:
                qc.barrier()

    # Final rotation layer (only RY)
    if not skip_final_rotation_layer:
        final_layer_idx = reps * (1 + int(hybrid))
        _rotation_layer(final_layer_idx, "ry")

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesExtended(QuantumCircuit):
    """Class‑style wrapper for the extended RealAmplitudes ansatz.

    The constructor signature mirrors :func:`real_amplitudes_extended`,
    exposing all the same knobs while providing a ``QuantumCircuit`` subclass
    that can be instantiated directly.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    reps : int, default=1
        Number of repetitions.
    entanglement : str or sequence or callable, default="full"
        Entanglement specification.
    skip_final_rotation_layer : bool, default=False
        Skip the final RY layer.
    insert_barriers : bool, default=False
        Insert barriers.
    parameter_prefix : str, default="theta"
        Prefix for parameters.
    hybrid : bool, default=False
        Include an RZ rotation block after each RY block.
    entangler_schedule : callable, optional
        Custom entangler schedule per repetition.
    name : str, default="RealAmplitudesExtended"
        Circuit name.

    Attributes
    ----------
    input_params : ParameterVector
        The parameters that feed the ansatz.
    num_rot_layers : int
        Number of rotation layers (including hybrid layers).
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        *,
        hybrid: bool = False,
        entangler_schedule: Optional[Callable[[int, int], Sequence[Tuple[int, int]]]] = None,
        name: str = "RealAmplitudesExtended",
    ) -> None:
        built = real_amplitudes_extended(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            hybrid=hybrid,
            entangler_schedule=entangler_schedule,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
