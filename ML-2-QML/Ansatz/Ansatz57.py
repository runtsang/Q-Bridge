"""
RealAmplitudesExtended ansatz builder.

This module defines a new parameterized ansatz that augments the
canonical RealAmplitudes circuit.  The extension adds:
  * **Hybrid rotation blocks** – RY followed by optional SX (or RZ) rotations.
  * **Depth‑controlled adaptive layer** – a user‑provided callable that can
    generate a new rotation depth per circuit layer.
  * **Entanglement schedule** – a user‑defined sequence of CX pairs that can
    be static, dynamic or a function of the circuit depth.
  * **Optional barriers** and **parameter prefixing** remain available.

The design keeps the original intuition: a stack of single‑qubit
rotations followed by two‑qubit entanglers, but it now offers richer
expressivity for research experiments.

Author: Qiskit Quantum Variational Circuits Team
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = ["RealAmplitudesExtended", "real_amplitudes_extended"]


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
    depth: int,
) -> List[Tuple[int, int]]:
    """Return entanglement pairs for a given depth.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the entanglement pattern.  ``"full"``, ``"linear"``, and
        ``"circular"`` are supported as static patterns.  If a callable is
        supplied it must accept a single integer argument ``depth`` and return a
        sequence of qubit pairs that will be used for that depth.
    depth : int
        Current entanglement depth index.

    Returns
    -------
    List[Tuple[int, int]]
        A list of (control, target) qubit indices that will receive a CX gate.

    Raises
    ------
    ValueError
        If an invalid pair is supplied or a pair references an out-of-range
        qubit.
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
        pairs = list(entanglement(depth))
        pairs = [(int(i), int(j)) for i, j in pairs]
    else:
        pairs = [(int(i), int(j)) for i, j in entanglement]

    for i, j in pairs:
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
    use_sx: bool = False,
    use_rz: bool = False,
    adaptive_depth_fn: Callable[[int], int] | None = None,
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended RealAmplitudes ansatz.

    The circuit is a stack of single‑qubit rotation layers followed by
    two‑qubit CX entanglers.  Each rotation layer can optionally include
    SX or RZ gates, and the number of sub‑layers per repetition can be
    adapted via ``adaptive_depth_fn``.  Entanglement can be static or
    supplied as a depth‑dependent callable.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int
        Number of repetition blocks.  Each block may contain a variable
        number of sub‑rotation layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement schedule.  If a callable is supplied it receives the
        current depth index and should return a sequence of (control,
        target) pairs.
    skip_final_rotation_layer : bool
        When ``True`` the final rotation block after the last
        entanglement is omitted.
    insert_barriers : bool
        Insert a barrier after each rotation and entanglement block.
    parameter_prefix : str
        Prefix for parameter names.
    use_sx : bool
        Append an SX rotation after each RY in the rotation block.
    use_rz : bool
        Append an RZ rotation after each RY in the rotation block.
    adaptive_depth_fn : Callable[[int], int] | None
        Function that returns the number of sub‑rotation layers for a given
        repetition index.  If ``None`` the depth is fixed to 1.
    name : str | None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if adaptive_depth_fn is not None and not callable(adaptive_depth_fn):
        raise ValueError("adaptive_depth_fn must be callable or None.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesExtended")

    # Determine the depth of each repetition block.
    depths_per_rep = [adaptive_depth_fn(r) if adaptive_depth_fn else 1 for r in range(reps)]
    if adaptive_depth_fn is not None:
        for depth in depths_per_rep:
            if depth < 1:
                raise ValueError("adaptive_depth_fn must return a positive integer (>=1) for each repetition.")

    total_sub_layers = sum(depths_per_rep)

    # Total rotation blocks executed (including optional final rotation).
    total_rot_blocks = total_sub_layers + (0 if skip_final_rotation_layer else 1)

    # Allocate parameter vectors.
    ry_params = ParameterVector(f"{parameter_prefix}_ry", total_rot_blocks * n)
    sx_params = ParameterVector(f"{parameter_prefix}_sx", total_rot_blocks * n) if use_sx else None
    rz_params = ParameterVector(f"{parameter_prefix}_rz", total_rot_blocks * n) if use_rz else None

    depth_index = 0
    for r in range(reps):
        sub_layers = depths_per_rep[r]
        for _ in range(sub_layers):
            # Rotation block.
            base = depth_index * n
            for q in range(n):
                qc.ry(ry_params[base + q], q)
            if use_sx:
                for q in range(n):
                    qc.sx(sx_params[base + q], q)
            if use_rz:
                for q in range(n):
                    qc.rz(rz_params[base + q], q)
            if insert_barriers:
                qc.barrier()
            # Entanglement.
            pairs = _resolve_entanglement(n, entanglement, depth_index)
            for i, j in pairs:
                qc.cx(i, j)
            if insert_barriers:
                qc.barrier()
            depth_index += 1

    # Final rotation layer (if requested).
    if not skip_final_rotation_layer:
        base = depth_index * n
        for q in range(n):
            qc.ry(ry_params[base + q], q)
        if use_sx:
            for q in range(n):
                qc.sx(sx_params[base + q], q)
        if use_rz:
            for q in range(n):
                qc.rz(rz_params[base + q], q)

    qc.input_params = [ry_params]
    if use_sx:
        qc.input_params.append(sx_params)
    if use_rz:
        qc.input_params.append(rz_params)

    qc.num_rot_layers = total_rot_blocks
    return qc


class RealAmplitudesExtended(QuantumCircuit):
    """Convenient subclass of :class:`~qiskit.QuantumCircuit` for the
    extended RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    reps : int
        Number of repetition blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.
    skip_final_rotation_layer : bool
        Skip the final rotation layer.
    insert_barriers : bool
        Insert barriers after each block.
    parameter_prefix : str
        Prefix for parameter names.
    use_sx : bool
        Append SX rotations.
    use_rz : bool
        Append RZ rotations.
    adaptive_depth_fn : Callable[[int], int] | None
        Function to compute sub‑layer depth per repetition.
    name : str
        Circuit name.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        use_sx: bool = False,
        use_rz: bool = False,
        adaptive_depth_fn: Callable[[int], int] | None = None,
        name: str = "RealAmplitudesExtended",
    ) -> None:
        built = real_amplitudes_extended(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            use_sx=use_sx,
            use_rz=use_rz,
            adaptive_depth_fn=adaptive_depth_fn,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params
        self.num_rot_layers = built.num_rot_layers

        # Store configuration for introspection.
        self.reps = reps
        self.entanglement = entanglement
        self.skip_final_rotation_layer = skip_final_rotation_layer
        self.insert_barriers = insert_barriers
        self.parameter_prefix = parameter_prefix
        self.use_sx = use_sx
        self.use_rz = use_rz
        self.adaptive_depth_fn = adaptive_depth_fn
