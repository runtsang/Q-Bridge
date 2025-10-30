"""RealAmplitudesExtended ansatz builder (RY/RZ + configurable entanglers)."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
    depth: int | None = None,
) -> List[Tuple[int, int]]:
    """
    Resolve entanglement pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the entanglement topology.
    depth : int | None, optional
        If provided, overrides the topology by connecting each qubit to
        the next ``depth`` neighbours in a cyclic fashion.

    Returns
    -------
    List[Tuple[int, int]]
        List of two‑qubit pairs.
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
        elif entanglement == "pairwise":
            pairs = [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
        else:
            raise ValueError(f"Unknown entanglement string: {entanglement!r}")
    elif callable(entanglement):
        pairs = list(entanglement(num_qubits))
    else:
        pairs = [(int(i), int(j)) for (i, j) in entanglement]
        for (i, j) in pairs:
            if i == j:
                raise ValueError("Entanglement pairs must connect distinct qubits.")
            if not (0 <= i < num_qubits and 0 <= j < num_qubits):
                raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")

    if depth is not None:
        if not (1 <= depth <= num_qubits - 1):
            raise ValueError("entanglement_depth must be between 1 and num_qubits-1.")
        depth_pairs: set[Tuple[int, int]] = set()
        for i in range(num_qubits):
            for d in range(1, depth + 1):
                j = (i + d) % num_qubits
                a, b = sorted((i, j))
                depth_pairs.add((a, b))
        pairs = list(depth_pairs)

    # Ensure no duplicate pairs
    return sorted(set(pairs))


def real_amplitudes_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    entanglement_depth: int | None = None,
    entangler: str = "cx",
    layer_type: str = "real",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended RealAmplitudes-style ``QuantumCircuit`` (RY/RZ + configurable CX/CZ/iSWAP entanglers).

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of rotation-entanglement blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement topology specification.
    entanglement_depth : int | None, default None
        Number of neighbours each qubit is entangled with (cyclically).  Overrides
        ``entanglement`` if provided.
    entangler : str, default "cx"
        Gate used for entanglement.  Options: "cx", "cz", "iswap".
    layer_type : str, default "real"
        Rotation layer type: "real" for RY only, "complex" for RY+RZ.
    skip_final_rotation_layer : bool, default False
        If True, omit the last rotation layer.
    insert_barriers : bool, default False
        Insert a barrier after each rotation or entanglement block.
    parameter_prefix : str, default "theta"
        Prefix for the generated parameters.
    name : str | None, default None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Notes
    -----
    * The circuit exposes ``input_params`` (the full :class:`~qiskit.circuit.ParameterVector`),
      ``num_rot_layers`` (int), ``params_per_qubit`` (int), ``layer_type`` (str),
      ``entangler`` (str), ``entanglement`` (spec), ``entanglement_depth`` (int),
      ``insert_barriers`` (bool), and ``skip_final_rotation_layer`` (bool) attributes.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if layer_type not in {"real", "complex"}:
        raise ValueError("layer_type must be'real' or 'complex'.")
    if entangler not in {"cx", "cz", "iswap"}:
        raise ValueError("entangler must be one of 'cx', 'cz', 'iswap'.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesExtended")

    pairs = _resolve_entanglement(n, entanglement, depth=entanglement_depth)

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params_per_qubit = 1 if layer_type == "real" else 2
    total_params = num_rot_layers * n * params_per_qubit
    params = ParameterVector(parameter_prefix, total_params)

    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * n * params_per_qubit
        for q in range(n):
            if layer_type == "real":
                qc.ry(params[base + q], q)
            else:  # complex
                qc.ry(params[base + 2 * q], q)
                qc.rz(params[base + 2 * q + 1], q)

    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            if entangler == "cx":
                qc.cx(i, j)
            elif entangler == "cz":
                qc.cz(i, j)
            else:  # iswap
                qc.iswap(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    # expose useful attributes
    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    qc.params_per_qubit = params_per_qubit  # type: ignore[attr-defined]
    qc.layer_type = layer_type  # type: ignore[attr-defined]
    qc.entangler = entangler  # type: ignore[attr-defined]
    qc.entanglement = entanglement  # type: ignore[attr-defined]
    qc.entanglement_depth = entanglement_depth  # type: ignore[attr-defined]
    qc.insert_barriers = insert_barriers  # type: ignore[attr-defined]
    qc.skip_final_rotation_layer = skip_final_rotation_layer  # type: ignore[attr-defined]
    qc.parameter_prefix = parameter_prefix  # type: ignore[attr-defined]

    return qc


class RealAmplitudesExtended(QuantumCircuit):
    """Convenience subclass of :class:`qiskit.circuit.QuantumCircuit` implementing the extended RealAmplitudes ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        entanglement_depth: int | None = None,
        entangler: str = "cx",
        layer_type: str = "real",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesExtended",
    ) -> None:
        built = real_amplitudes_extended(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            entanglement_depth=entanglement_depth,
            entangler=entangler,
            layer_type=layer_type,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.params_per_qubit = built.params_per_qubit  # type: ignore[attr-defined]
        self.layer_type = built.layer_type  # type: ignore[attr-defined]
        self.entangler = built.entangler  # type: ignore[attr-defined]
        self.entanglement = built.entanglement  # type: ignore[attr-defined]
        self.entanglement_depth = built.entanglement_depth  # type: ignore[attr-defined]
        self.insert_barriers = built.insert_barriers  # type: ignore[attr-defined]
        self.skip_final_rotation_layer = built.skip_final_rotation_layer  # type: ignore[attr-defined]
        self.parameter_prefix = built.parameter_prefix  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesExtended", "real_amplitudes_extended"]
