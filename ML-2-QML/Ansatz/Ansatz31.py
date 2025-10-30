"""Extended RealAmplitudes ansatz with configurable rotation and entanglement gates."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]],
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

    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _validate_pairs(
    pairs: Sequence[Tuple[int, int]],
    num_qubits: int,
) -> List[Tuple[int, int]]:
    """Validate entanglement pairs for correctness."""
    validated = []
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
        validated.append((int(i), int(j)))
    return validated


def real_amplitudes_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]] = "full",
    entanglement_gate: str = "cx",
    rotation_gate: str = "ry",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended RealAmplitudes-style ``QuantumCircuit``.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of rotation/entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.  If a callable, it is treated as a layer‑wise
        schedule and must accept the layer index.
    entanglement_gate : str, {"cx", "cz", "rzr"}, default "cx"
        Two‑qubit gate used for entanglement.
    rotation_gate : str, {"ry", "rz", "rx"}, default "ry"
        Single‑qubit rotation axis.
    skip_final_rotation_layer : bool, default False
        If True, the final rotation layer is omitted.
    insert_barriers : bool, default False
        Whether to insert barriers between layers for visual clarity.
    parameter_prefix : str, default "theta"
        Prefix for rotation parameters.  For RZZ entanglement a suffix ``_ent`` is
        appended.
    name : str, optional
        Name of the constructed circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesExtended")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    rot_params = ParameterVector(parameter_prefix, num_rot_layers * n)

    # Choose rotation function
    if rotation_gate == "ry":
        rot_func = qc.ry
    elif rotation_gate == "rz":
        rot_func = qc.rz
    elif rotation_gate == "rx":
        rot_func = qc.rx
    else:
        raise ValueError(f"Unsupported rotation_gate: {rotation_gate!r}")

    # Choose entanglement function
    if entanglement_gate == "cx":
        ent_func = lambda i, j: qc.cx(i, j)
    elif entanglement_gate == "cz":
        ent_func = lambda i, j: qc.cz(i, j)
    elif entanglement_gate == "rzr":
        # For RZZ we need a parameter per pair per layer
        if callable(entanglement):
            max_pairs = max(
                len(_validate_pairs(entanglement(layer), n)) for layer in range(reps)
            )
        else:
            max_pairs = len(_validate_pairs(_resolve_entanglement(n, entanglement), n))
        ent_params = ParameterVector(parameter_prefix + "_ent", reps * max_pairs)
    else:
        raise ValueError(f"Unsupported entanglement_gate: {entanglement_gate!r}")

    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * n
        for q in range(n):
            rot_func(rot_params[base + q], q)

    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()

        # Determine the entanglement pairs for this layer
        if callable(entanglement):
            pairs = _validate_pairs(entanglement(r), n)
        else:
            pairs = _validate_pairs(_resolve_entanglement(n, entanglement), n)

        if entanglement_gate == "rzr":
            # RZZ parameters are laid out layer‑wise with a fixed stride
            for idx, (i, j) in enumerate(pairs):
                param = ent_params[r * max_pairs + idx]
                qc.rzz(param, i, j)
        else:
            for i, j in pairs:
                ent_func(i, j)

        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    qc.input_params = rot_params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    if entanglement_gate == "rzr":
        qc.entanglement_params = ent_params  # type: ignore[attr-defined]
    return qc


class RealAmplitudesExtended(QuantumCircuit):
    """Class‑style wrapper for the extended RealAmplitudes ansatz.

    The constructor mirrors the functional interface and exposes the
    underlying parameter vectors as ``input_params`` and
    ``entanglement_params`` (if applicable).  The circuit can be reused,
    composed, or bound with parameters just like any Qiskit ``QuantumCircuit``.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]] = "full",
        entanglement_gate: str = "cx",
        rotation_gate: str = "ry",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesExtended",
    ) -> None:
        built = real_amplitudes_extended(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            entanglement_gate=entanglement_gate,
            rotation_gate=rotation_gate,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        if hasattr(built, "entanglement_params"):
            self.entanglement_params = built.entanglement_params  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesExtended", "real_amplitudes_extended"]
