"""Extended RealAmplitudes ansatz with hybrid layers and adaptive entanglement.

Key extensions:
  * Hybrid rotation layers (RY followed optionally by RZ) for increased expressivity.
  * Configurable entanglement schedules: string presets, custom lists, or per‑repetition callables.
  * Optional random entanglement for stochastic exploration.
  * Parameter sharing across repetitions is preserved; each layer receives its own parameter set.
  * Barrier insertion and final‑rotation skipping retained for compatibility with training pipelines.

The module exposes a convenience function `real_amplitudes_plus` and a subclass
`RealAmplitudesPlus` that behaves like Qiskit's `QuantumCircuit`.  Both share the
same name for seamless integration.
"""

from __future__ import annotations

import random
from typing import Callable, Iterable, List, Sequence, Tuple, Union

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
    """Return a list of two‑qubit pairs according to a simple entanglement spec."""
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
        if entanglement == "random":
            # deterministic random for reproducibility
            rng = random.Random(42)
            pairs = []
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    if rng.random() < 0.5:
                        pairs.append((i, j))
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


def real_amplitudes_plus(
    num_qubits: int,
    reps: int = 1,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ] = "full",
    entanglement_schedule: Callable[[int], Sequence[Tuple[int, int]]] | None = None,
    include_rz: bool = False,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended RealAmplitudes-style circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of rotation + entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the entanglement pattern.  Supported strings:
            ``"full"``, ``"linear"``, ``"circular"``, ``"random"``.
        If a callable or explicit list is supplied, it is used verbatim.
    entanglement_schedule : Callable[[int], Sequence[Tuple[int, int]]] | None, default None
        Optional per‑repetition schedule that overrides ``entanglement``.
        The callable receives the repetition index (starting at 0) and must
        return a list of (i, j) tuples for that repetition.
    include_rz : bool, default False
        If True, an RZ rotation is applied after each RY in every layer,
        doubling the number of parameters per qubit.
    skip_final_rotation_layer : bool, default False
        Whether to omit the final rotation layer (useful for variational
        circuits that end with a measurement or a different layer).
    insert_barriers : bool, default False
        Insert barriers between logical layers for better visualisation.
    parameter_prefix : str, default "theta"
        Prefix for the parameter vector names.
    name : str | None, default None
        Circuit name.  If omitted, defaults to ``"RealAmplitudesPlus"``.

    Returns
    -------
    QuantumCircuit
        The constructed variational circuit.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesPlus")

    # Determine total number of rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    # Each qubit receives one RY per rotation layer; optional RZ doubles parameters
    params_per_layer = n * (1 + int(include_rz))
    params = ParameterVector(parameter_prefix, num_rot_layers * params_per_layer)

    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * params_per_layer
        for q in range(n):
            qc.ry(params[base + q], q)
            if include_rz:
                qc.rz(params[base + n + q], q)

    # Resolve entanglement pairs per repetition
    if entanglement_schedule is None:
        base_pairs = _resolve_entanglement(n, entanglement)
        entanglement_pairs = [base_pairs] * reps
    else:
        entanglement_pairs = [
            [(int(i), int(j)) for (i, j) in entanglement_schedule(r)] for r in range(reps)
        ]

    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in entanglement_pairs[r]:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesPlus(QuantumCircuit):
    """Class‑style wrapper for the extended RealAmplitudes ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: Union[
            str,
            Sequence[Tuple[int, int]],
            Callable[[int], Sequence[Tuple[int, int]]],
        ] = "full",
        entanglement_schedule: Callable[[int], Sequence[Tuple[int, int]]] | None = None,
        include_rz: bool = False,
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesPlus",
    ) -> None:
        built = real_amplitudes_plus(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            entanglement_schedule=entanglement_schedule,
            include_rz=include_rz,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesPlus", "real_amplitudes_plus"]
