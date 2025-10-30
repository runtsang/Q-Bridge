"""Extended RealAmplitudes ansatz.

The module exposes a convenience function ``real_amplitudes_extended`` and a
class ``RealAmplitudesExtended`` that inherits from ``QuantumCircuit``.  It
keeps the original RY‑CX pattern but adds:

* **hybrid rotation layers** – apply RY, RX and RZ on each qubit.
* **custom entanglement schedules** – per‑repetition CX patterns via a callable.
* **depth control** – ``max_depth`` limits the number of repetitions.
* **parameter sharing** – reuse the same set of rotation parameters across
  all layers if desired.

These extensions provide a richer variational family while remaining
compatible with Qiskit workflows.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int, int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int, int], Sequence[Tuple[int, int]]]
        One of ``'full'``, ``'linear'``, ``'circular'`` or a custom sequence/callable.

    Returns
    -------
    List[Tuple[int, int]]
        List of distinct qubit pairs to be entangled with CX gates.

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
        pairs = list(entanglement(num_qubits, num_qubits))
        return [(int(i), int(j)) for (i, j) in pairs]

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
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int, int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    rotation_layer_type: str = "ry",
    entanglement_schedule: Callable[[int, int], Sequence[Tuple[int, int]]] | None = None,
    max_depth: int | None = None,
    share_parameters: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended RealAmplitudes-style ``QuantumCircuit``.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of rotation + entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int, int], Sequence[Tuple[int, int]]], default "full"
        Entanglement pattern for each repetition.  Can be a callable that
        receives (repetition_index, num_qubits) and returns a list of pairs.
    skip_final_rotation_layer : bool, default False
        If ``True`` the final rotation layer after the last entanglement is omitted.
    insert_barriers : bool, default False
        Insert a barrier after each rotation and entanglement block.
    parameter_prefix : str, default "theta"
        Prefix for the parameters in the ``ParameterVector``.
    rotation_layer_type : str, default "ry"
        Either ``"ry"`` (original) or ``"hybrid"`` (RY + RX + RZ per qubit).
    entanglement_schedule : Callable[[int, int], Sequence[Tuple[int, int]]], optional
        If supplied, overrides the entanglement pattern for each repetition.
    max_depth : int, optional
        Maximum number of repetitions to apply.  Overrides ``reps`` if smaller.
    share_parameters : bool, default False
        If ``True`` the same set of rotation parameters is reused across all layers.
    name : str | None, default None
        Name of the circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1, or if ``rotation_layer_type`` is unknown.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    if rotation_layer_type not in {"ry", "hybrid"}:
        raise ValueError(f"Unsupported rotation_layer_type: {rotation_layer_type!r}")

    if max_depth is not None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        reps = min(reps, max_depth)

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesExtended")

    # Determine rotation factor: 1 for RY only, 3 for hybrid RY-RX-RZ
    rotation_factor = 1 if rotation_layer_type == "ry" else 3
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter vector size
    if share_parameters:
        param_count = n * rotation_factor
    else:
        param_count = num_rot_layers * n * rotation_factor

    params = ParameterVector(parameter_prefix, param_count)

    def _rotation_layer(layer_idx: int) -> None:
        """Apply one rotation layer of the chosen type."""
        if share_parameters:
            base = 0
        else:
            base = layer_idx * n * rotation_factor

        for q in range(n):
            if rotation_layer_type == "ry":
                qc.ry(params[base + q], q)
            else:  # hybrid
                idx = base + q * rotation_factor
                qc.ry(params[idx], q)
                qc.rx(params[idx + 1], q)
                qc.rz(params[idx + 2], q)

    # Precompute entanglement pairs if static
    static_pairs = None
    if entanglement_schedule is None:
        static_pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        # Determine entanglement pairs for this repetition
        if entanglement_schedule is not None:
            pairs = _resolve_entanglement(
                n, entanglement_schedule(r, n)
            )
        else:
            pairs = static_pairs  # type: ignore[assignment]
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    qc.rotation_layer_type = rotation_layer_type  # type: ignore[attr-defined]
    return qc


class RealAmplitudesExtended(QuantumCircuit):
    """Convenience class wrapper for the extended RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of rotation + entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int, int], Sequence[Tuple[int, int]]], default "full"
        Entanglement pattern for each repetition.
    skip_final_rotation_layer : bool, default False
        If ``True`` the final rotation layer after the last entanglement is omitted.
    insert_barriers : bool, default False
        Insert a barrier after each rotation and entanglement block.
    parameter_prefix : str, default "theta"
        Prefix for the parameters in the ``ParameterVector``.
    rotation_layer_type : str, default "ry"
        Either ``"ry"`` (original) or ``"hybrid"`` (RY + RX + RZ per qubit).
    entanglement_schedule : Callable[[int, int], Sequence[Tuple[int, int]]], optional
        If supplied, overrides the entanglement pattern for each repetition.
    max_depth : int, optional
        Maximum number of repetitions to apply.  Overrides ``reps`` if smaller.
    share_parameters : bool, default False
        If ``True`` the same set of rotation parameters is reused across all layers.
    name : str, default "RealAmplitudesExtended"
        Name of the circuit.

    Notes
    -----
    The constructor builds the ansatz via :func:`real_amplitudes_extended` and then
    composes the resulting circuit into ``self``.  All attributes from the
    underlying circuit (``input_params``, ``num_rot_layers`` etc.) are exposed.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int, int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        rotation_layer_type: str = "ry",
        entanglement_schedule: Callable[[int, int], Sequence[Tuple[int, int]]] | None = None,
        max_depth: int | None = None,
        share_parameters: bool = False,
        name: str = "RealAmplitudesExtended",
    ) -> None:
        built = real_amplitudes_extended(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            rotation_layer_type=rotation_layer_type,
            entanglement_schedule=entanglement_schedule,
            max_depth=max_depth,
            share_parameters=share_parameters,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.rotation_layer_type = built.rotation_layer_type  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesExtended", "real_amplitudes_extended"]
