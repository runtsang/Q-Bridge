"""
Extended RealAmplitudes ansatz with flexible entanglement, RZZ layers, and rotation alternation.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification:
        * ``"full"``   – all possible pairs
        * ``"linear"`` – nearest neighbour chain
        * ``"circular"`` – linear + final connection
        * ``"star"``   – qubit 0 connected to all others
        * ``"random"`` – deterministic random shuffle of all pairs
        * ``"none"``   – no entanglement
        * custom sequence of pairs
        * callable returning a sequence

    Returns
    -------
    List[Tuple[int, int]]
        List of distinct qubit pairs.

    Raises
    ------
    ValueError
        If an invalid entanglement specification is provided or
        a pair contains identical qubits or indices out of range.
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
            return [(0, i) for i in range(1, num_qubits)]
        if entanglement == "random":
            import random
            rng = random.Random(42)  # deterministic seed
            pairs = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
            rng.shuffle(pairs)
            return pairs
        if entanglement == "none":
            return []
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


# --------------------------------------------------------------------------- #
# Ansatz builder
# --------------------------------------------------------------------------- #
def real_amplitudes(
    num_qubits: int,
    reps: int = 1,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ] = "full",
    entanglement_depth: int = 1,
    rotation_type: str = "ry",
    insert_rzz: bool = False,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended RealAmplitudes-style circuit.

    The circuit consists of ``reps`` repetition blocks, each containing a
    rotation layer followed by ``entanglement_depth`` entangling stages.
    Optionally, an RZZ entangling layer may be inserted after each CX
    stage. The rotation layer can use RY, RZ, or an alternating RY/RZ pattern.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of repetition blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement pattern or custom specification.
    entanglement_depth : int, default 1
        Number of entangling stages per repetition.  ``entanglement_depth=2``
        adds a second CX layer per repetition, increasing expressive power.
    rotation_type : str, default "ry"
        Rotation type for the single‑qubit layers:
        * ``"ry"`` – all RY gates
        * ``"rz"`` – all RZ gates
        * ``"ryrz"`` – alternate RY and RZ per qubit
    insert_rzz : bool, default False
        If ``True``, an RZZ gate is applied after each CX pair,
        using a dedicated parameter vector.
    skip_final_rotation_layer : bool, default False
        Omit the final rotation layer after the last repetition.
    insert_barriers : bool, default False
        Insert barriers between layers for visual clarity.
    parameter_prefix : str, default "theta"
        Prefix for the rotation parameters.
    name : str | None, default None
        Circuit name; if ``None`` defaults to ``"RealAmplitudes"``.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1, or if an invalid ``rotation_type``
        or ``entanglement_depth`` is provided.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if entanglement_depth < 1:
        raise ValueError("entanglement_depth must be >= 1.")
    if rotation_type not in {"ry", "rz", "ryrz"}:
        raise ValueError(f"Unsupported rotation_type: {rotation_type!r}")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudes")

    # Determine number of rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    # Optional RZZ parameters
    pairs = _resolve_entanglement(n, entanglement)
    num_pairs = len(pairs)
    rzz_params: ParameterVector | None = None
    if insert_rzz and num_pairs > 0:
        rzz_params = ParameterVector(f"{parameter_prefix}_rz", num_rot_layers * num_pairs)

    def _rotation_layer(layer_idx: int) -> None:
        """
        Apply a single‑qubit rotation layer of the chosen type.
        """
        base = layer_idx * n
        for q in range(n):
            if rotation_type == "ry":
                qc.ry(params[base + q], q)
            elif rotation_type == "rz":
                qc.rz(params[base + q], q)
            else:  # ryrz
                gate = qc.ry if q % 2 == 0 else qc.rz
                gate(params[base + q], q)

    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        # Entanglement depth loop
        for d in range(entanglement_depth):
            for (i, j) in pairs:
                qc.cx(i, j)
                if insert_rzz:
                    idx = r * num_pairs + d * num_pairs + pairs.index((i, j))
                    qc.rzz(rzz_params[idx], i, j)
            if insert_barriers:
                qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    if insert_rzz:
        qc.rzz_params = rzz_params  # type: ignore[attr-defined]
    return qc


# --------------------------------------------------------------------------- #
# Class wrapper
# --------------------------------------------------------------------------- #
class RealAmplitudes(QuantumCircuit):
    """
    Class‑style wrapper for the extended RealAmplitudes ansatz.

    The constructor forwards all arguments to :func:`real_amplitudes`
    and composes the resulting circuit, exposing the same
    ``input_params`` and ``num_rot_layers`` attributes.
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
        entanglement_depth: int = 1,
        rotation_type: str = "ry",
        insert_rzz: bool = False,
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudes",
    ) -> None:
        built = real_amplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            entanglement_depth=entanglement_depth,
            rotation_type=rotation_type,
            insert_rzz=insert_rzz,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        if insert_rzz:
            self.rzz_params = built.rzz_params  # type: ignore[attr-defined]


__all__ = ["RealAmplitudes", "real_amplitudes"]
