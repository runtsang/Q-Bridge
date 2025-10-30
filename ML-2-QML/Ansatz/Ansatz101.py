"""
RealAmplitudesAlternatingExtended: A richer, Qiskit‑compatible ansatz that builds upon the original alternating‑rotation RealAmplitudes variant.

Design highlights
-----------------
* **Depth‑control** – the `depth` argument controls the number of full rotation‑entanglement blocks.
* **Mid‑layer entanglement** – when `use_mid_entanglement=True` an extra entanglement block is inserted *after* each rotation.
* **Hybrid rotation** – when `use_hybrid_rotation=True` each rotation layer applies an RX/RY alternation *and* a global RZ gate on every qubit.
* **Entanglement schedule** – accepts the same flexible specification as the seed (`str`, `Sequence[Tuple[int, int]]` or `Callable[[int], Sequence[Tuple[int, int]]]`).
* **Barrier support** – optional barriers can be inserted after each rotation or entanglement block.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve the entanglement specification into a list of qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification. Accepted strings are ``"full"``, ``"linear"``, and ``"circular"``.
        Alternatively, a list of pairs or a callable returning such a list can be supplied.

    Returns
    -------
    List[Tuple[int, int]]
        A list of distinct qubit index pairs.

    Raises
    ------
    ValueError
        If an unknown string is supplied, or if a pair references an out‑of‑range qubit.
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

    # assume a sequence of tuples
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def real_amplitudes_alternating_extended(
    num_qubits: int,
    depth: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    use_mid_entanglement: bool = False,
    use_hybrid_rotation: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a RealAmplitudes‑style ansatz with alternating RX/RY rotations and optional enhancements.

    The circuit consists of ``depth`` full rotation‑entanglement blocks.  Each block
    contains an alternating rotation layer (even layers use RY, odd layers use RX).
    When ``use_hybrid_rotation=True`` an additional RZ gate is applied to every qubit
    *after* the alternating rotation.  If ``use_mid_entanglement=True`` an extra
    entanglement block is inserted *after* the rotation but *before* the standard
    entanglement block.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    depth : int, default 1
        Number of rotation‑entanglement blocks to include.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Specification of two‑qubit entanglement.  Accepted strings are ``"full"``,
        ``"linear"``, and ``"circular"``.  Alternatively, provide a custom list of
        qubit pairs or a callable returning such a list.
    skip_final_rotation_layer : bool, default False
        If ``True`` the final rotation layer is omitted, mirroring the behaviour of
        the original seed ansatz.
    insert_barriers : bool, default False
        If ``True`` a barrier is inserted after each rotation and entanglement block.
    use_mid_entanglement : bool, default False
        If ``True`` an additional entanglement block is inserted after each rotation
        layer, before the main entanglement block.
    use_hybrid_rotation : bool, default False
        If ``True`` each rotation layer applies an RX/RY alternation *and* a global
        RZ gate on every qubit, increasing expressivity.
    parameter_prefix : str, default "theta"
        Prefix for the parameter vector.
    name : str | None, default None
        Optional name for the circuit; defaults to ``"RealAmplitudesAlternatingExtended"``.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.  The returned circuit has attributes
        ``input_params`` (the :class:`ParameterVector` instance) and
        ``num_rot_layers`` (the total number of rotation layers applied).

    Raises
    ------
    ValueError
        If ``num_qubits`` or ``depth`` are less than 1.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if depth < 1:
        raise ValueError("depth must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingExtended")

    num_rot_layers = depth if skip_final_rotation_layer else depth + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rotation(layer: int) -> None:
        """Apply an alternating rotation layer (and optional RZ)."""
        base = layer * n
        for q in range(n):
            if layer % 2 == 0:
                qc.ry(params[base + q], q)
            else:
                qc.rx(params[base + q], q)
        if use_hybrid_rotation:
            # Global RZ after the alternating rotation
            for q in range(n):
                qc.rz(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(depth):
        _rotation(r)
        if insert_barriers:
            qc.barrier()
        if use_mid_entanglement:
            for (i, j) in pairs:
                qc.cx(i, j)
            if insert_barriers:
                qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation(depth)
        if insert_barriers:
            qc.barrier()

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingExtended(QuantumCircuit):
    """
    Qiskit wrapper class for :func:`real_amplitudes_alternating_extended`.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    depth : int, default 1
        Number of rotation‑entanglement blocks to include.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Specification of two‑qubit entanglement.
    skip_final_rotation_layer : bool, default False
        If ``True`` the final rotation layer is omitted.
    insert_barriers : bool, default False
        Insert barriers between blocks.
    use_mid_entanglement : bool, default False
        Insert an additional entanglement block after each rotation.
    use_hybrid_rotation : bool, default False
        Apply a global RZ gate following the alternating rotation.
    parameter_prefix : str, default "theta"
        Prefix for the parameter vector.
    name : str, default "RealAmplitudesAlternatingExtended"
        Optional circuit name.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        use_mid_entanglement: bool = False,
        use_hybrid_rotation: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingExtended",
    ) -> None:
        built = real_amplitudes_alternating_extended(
            num_qubits,
            depth,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            use_mid_entanglement,
            use_hybrid_rotation,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = [
    "RealAmplitudesAlternatingExtended",
    "real_amplitudes_alternating_extended",
]
