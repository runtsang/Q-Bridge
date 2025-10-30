"""
RealAmplitudes variant with alternating RY/RX rotation layers and optional
parameter sharing across qubits (controlled modification).

This module exposes:
    - `real_amplitudes_alternating_controlled_modification` – convenience
      constructor returning a `QuantumCircuit`.
    - `RealAmplitudesAlternatingControlledModification` – a subclass of
      `QuantumCircuit` that builds the ansatz on initialization.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
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


def real_amplitudes_alternating_controlled_modification(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    share_params: bool = True,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a RealAmplitudes ansatz with alternating RY/RX layers and optional
    parameter sharing across qubits.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit. Must be >= 1.
    reps : int, default 1
        Number of entanglement layers. Each repetition adds one rotation layer
        per qubit (or a shared layer if ``share_params=True``), one entanglement
        block, and optionally a barrier.
    entanglement : str or sequence or callable, default "full"
        Specification of which qubit pairs to entangle. Accepted strings are
        "full", "linear", and "circular".  Alternatively a sequence of
        (i, j) tuples or a callable returning such a sequence.
    skip_final_rotation_layer : bool, default False
        If True, the final rotation layer after the last entanglement block
        is omitted.  This matches the behaviour of the seed ansatz.
    insert_barriers : bool, default False
        If True, insert barriers after each rotation and entanglement block.
    parameter_prefix : str, default "theta"
        Prefix for automatically generated rotation parameters.
    share_params : bool, default True
        If True, a single rotation parameter per layer is used and applied to
        every qubit in that layer.  If False, each qubit receives its own
        independent parameter (original behaviour).
    name : str, optional
        Name of the constructed circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.  The circuit exposes two attributes:
        ``input_params`` (a :class:`ParameterVector`) and ``num_rot_layers``
        (int).

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1 or ``share_params`` is not a bool.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if not isinstance(share_params, bool):
        raise ValueError("share_params must be a bool.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingControlled")

    # Determine number of rotation layers: one per repetition plus optional final layer.
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter vector: one per layer if sharing, otherwise one per qubit per layer.
    param_len = num_rot_layers * (1 if share_params else n)
    params = ParameterVector(parameter_prefix, param_len)

    def _rot(layer: int) -> None:
        """
        Append the rotation layer for the given repetition index.
        When ``share_params`` is True, the same angle is applied to all qubits.
        """
        base = layer * (1 if share_params else n)
        if layer % 2 == 0:
            # RY rotations
            if share_params:
                angle = params[base]
                for q in range(n):
                    qc.ry(angle, q)
            else:
                for q in range(n):
                    qc.ry(params[base + q], q)
        else:
            # RX rotations
            if share_params:
                angle = params[base]
                for q in range(n):
                    qc.rx(angle, q)
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

    # Attach metadata for downstream tooling.
    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingControlledModification(QuantumCircuit):
    """
    QuantumCircuit subclass that builds the RealAmplitudes alternating ansatz
    with optional parameter sharing across qubits.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of entanglement layers.
    entanglement : str or sequence or callable, default "full"
        Entanglement specification.
    skip_final_rotation_layer : bool, default False
        Skip the final rotation layer if True.
    insert_barriers : bool, default False
        Insert barriers after each block if True.
    parameter_prefix : str, default "theta"
        Prefix for rotation parameters.
    share_params : bool, default True
        Whether to share a single rotation parameter per layer.
    name : str, default "RealAmplitudesAlternatingControlledModification"
        Circuit name.

    Notes
    -----
    The instance exposes the same ``input_params`` and ``num_rot_layers``
    attributes as the convenience constructor.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        share_params: bool = True,
        name: str = "RealAmplitudesAlternatingControlledModification",
    ) -> None:
        built = real_amplitudes_alternating_controlled_modification(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            share_params,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = [
    "RealAmplitudesAlternatingControlledModification",
    "real_amplitudes_alternating_controlled_modification",
]
