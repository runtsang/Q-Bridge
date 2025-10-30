"""RealAmplitudesCZExtended – a depth‑controlled, hybrid‑layer ansatz.

The design builds upon the original RealAmplitudesCZ by
  * adding a second rotation layer (RZ) per repetition for extra expressivity.
  * supporting a “parity” entanglement mode that couples qubits of the same parity.
  * exposing configurable mid‑layer barriers for clearer circuit structure.
  * keeping the same interface with a convenience constructor and a subclass.
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
    """Resolve an entanglement specification into a list of qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.  Supported strings are
        ``"full"``, ``"linear"``, ``"circular"``, and ``"parity"``.
        A sequence of pairs can be supplied directly, or a callable
        that returns such a sequence given *num_qubits*.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of unique qubit pairs.

    Raises
    ------
    ValueError
        If an unknown string is supplied or if a pair references an
        out‑of‑range qubit or the same qubit twice.
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
        if entanglement == "parity":
            # Entangle qubits of equal parity (even-even, odd-odd)
            pairs: List[Tuple[int, int]] = []
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    if (i % 2) == (j % 2):
                        pairs.append((i, j))
            return pairs
        raise ValueError(f"Unknown entanglement string: {entanglement!r}")

    if callable(entanglement):
        pairs = list(entanglement(num_qubits))
        return [(int(i), int(j)) for i, j in pairs]

    # Treat as explicit list of pairs
    pairs = [(int(i), int(j)) for i, j in entanglement]
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ] = "full",
    skip_final_rotation_layer: bool = False,
    mid_layer_barrier: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build a depth‑controlled hybrid rotation‑entanglement ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of repetition cycles.  Each cycle consists of an
        Ry rotation layer, an optional entanglement block, and an
        RZ rotation layer.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern.  ``"full"``, ``"linear"``, ``"circular"``,
        and ``"parity"`` are supported as strings.  A list of pairs or
        a callable returning such a list can also be supplied.
    skip_final_rotation_layer : bool, default False
        If True, the final Ry layer after the last repetition is omitted.
    mid_layer_barrier : bool, default False
        If True, insert a barrier between the Ry and entanglement
        layers and another between the entanglement and RZ layers.
    insert_barriers : bool, default False
        If True, insert a barrier after each entanglement block and
        after each RZ layer for visual clarity.
    parameter_prefix : str, default "theta"
        Prefix used for the ParameterVector names.
    name : str or None, default None
        Name of the resulting QuantumCircuit.  If *None*, a default
        name derived from the class name is used.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit, with ``input_params`` and
        ``num_rot_layers`` attributes attached for convenience.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    # Resolve entanglement pairs once; the same pattern is reused for each layer.
    pairs = _resolve_entanglement(n, entanglement)

    # Calculate total number of parameters.
    # Each repetition contributes 2*n parameters (Ry + RZ).
    # Optionally a final Ry layer adds n parameters.
    total_params = reps * 2 * n + (0 if skip_final_rotation_layer else n)
    params = ParameterVector(parameter_prefix, total_params)

    # Helper functions to apply rotation layers.
    def _ry_layer(layer: int) -> None:
        base = layer * 2 * n
        for q in range(n):
            qc.ry(params[base + q], q)

    def _rz_layer(layer: int) -> None:
        base = layer * 2 * n + n
        for q in range(n):
            qc.rz(params[base + q], q)

    # Build the circuit.
    for r in range(reps):
        _ry_layer(r)
        if mid_layer_barrier:
            qc.barrier()
        for i, j in pairs:
            qc.cz(i, j)
        if mid_layer_barrier:
            qc.barrier()
        _rz_layer(r)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        base = reps * 2 * n
        for q in range(n):
            qc.ry(params[base + q], q)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = reps  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZExtended(QuantumCircuit):
    """Convenience wrapper for the extended CZ‑entangling ansatz.

    This class simply builds the circuit via :func:`real_amplitudes_cz_extended`
    and then composes it into a new ``QuantumCircuit`` instance.  It exposes
    the same public attributes as the base function.
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
        skip_final_rotation_layer: bool = False,
        mid_layer_barrier: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            mid_layer_barrier,
            insert_barriers,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]
