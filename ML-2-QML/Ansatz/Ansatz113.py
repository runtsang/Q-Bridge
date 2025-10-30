"""Extended RealAmplitudes ansatz with additional expressive layers."""
from __future__ import annotations

from typing import Callable, Iterable, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[
        str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]
    ],
) -> list[Tuple[int, int]]:
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


def _get_entangler_gate(entangler: str):
    """Map entangler name to the corresponding Qiskit gate function."""
    mapping = {
        "cx": QuantumCircuit.cx,
        "cnot": QuantumCircuit.cx,
        "cz": QuantumCircuit.cz,
        "crx": QuantumCircuit.crx,
        "crz": QuantumCircuit.crz,
    }
    if entangler not in mapping:
        raise ValueError(f"Unsupported entangler: {entangler!r}")
    return mapping[entangler]


def extended_real_amplitudes(
    *,
    num_qubits: int,
    reps: int = 1,
    entanglement: Union[
        str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]
    ] = "full",
    entangler: str = "cx",
    rotation: str = "ry",
    entanglement_depth: int = 1,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_sharing: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended RealAmplitudes-style circuit.

    The circuit consists of alternating rotation and entanglement layers.  Each
    rotation layer applies either RY, RZ, or both RY and RZ rotations to every
    qubit.  The entanglement layer can use one of several two‑qubit gates
    (CX, CZ, CRX, CRZ), and the same pairs can be applied multiple times per
    layer via ``entanglement_depth``.  Parameters may be shared across qubits
    within a layer using ``parameter_sharing``.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.  Must be >= 1.
    reps : int, default 1
        Number of full rotation‑entanglement repeats.  If
        ``skip_final_rotation_layer`` is False, an additional rotation layer
        is added at the end.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.  Accepted strings are ``"full"``,
        ``"linear"``, and ``"circular"``.  Custom sequences or callables may
        also be supplied.
    entangler : str, default "cx"
        Two‑qubit gate used for entanglement.  Supported values are
        ``"cx"``, ``"cnot"``, ``"cz"``, ``"crx"``, and ``"crz"``.
    rotation : str, default "ry"
        Rotation type applied to each qubit.  Acceptable values are:
        ``"ry"``, ``"rz"``, and ``"ryrz"``, which applies both RY and RZ.
    entanglement_depth : int, default 1
        Number of times each entanglement pair is applied per layer.
    skip_final_rotation_layer : bool, default False
        If True, the final rotation layer after the last entanglement is omitted.
    insert_barriers : bool, default False
        If True, a barrier is inserted after each rotation and entanglement
        sub‑layer to aid circuit visualization and debugging.
    parameter_sharing : bool, default False
        If True, a single parameter per rotation type per layer is shared
        across all qubits.  This reduces the total number of parameters.
    parameter_prefix : str, default "theta"
        Prefix used for rotation parameters.  For ``"ryrz"`` a second
        ``_rz`` suffix is appended automatically.
    name : str | None, default None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.  The circuit exposes the following
        attributes for convenience:

        * ``input_params`` – list of :class:`ParameterVector` objects
          containing all parameters in the order they were created.
        * ``num_rot_layers`` – number of rotation layers added.
        * ``rotation_type`` – the rotation string passed to the function.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    if reps < 1:
        raise ValueError("reps must be >= 1.")

    if entanglement_depth < 1:
        raise ValueError("entanglement_depth must be >= 1.")

    if rotation not in {"ry", "rz", "ryrz"}:
        raise ValueError(f"Unsupported rotation type: {rotation!r}")

    entangler_gate = _get_entangler_gate(entangler)
    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "ExtendedRealAmplitudes")

    # Determine number of rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter vectors
    if rotation == "ryrz":
        if parameter_sharing:
            params_ry = ParameterVector(parameter_prefix, num_rot_layers)
            params_rz = ParameterVector(f"{parameter_prefix}_rz", num_rot_layers)
        else:
            params_ry = ParameterVector(parameter_prefix, num_rot_layers * n)
            params_rz = ParameterVector(f"{parameter_prefix}_rz", num_rot_layers * n)
    else:
        param_name = parameter_prefix if rotation == "ry" else f"{parameter_prefix}_rz"
        if parameter_sharing:
            params = ParameterVector(param_name, num_rot_layers)
        else:
            params = ParameterVector(param_name, num_rot_layers * n)

    # Helper to apply rotation layer
    def _rotation_layer(layer_idx: int) -> None:
        if rotation == "ryrz":
            if parameter_sharing:
                for q in range(n):
                    qc.ry(params_ry[layer_idx], q)
                    qc.rz(params_rz[layer_idx], q)
            else:
                base = layer_idx * n
                for q in range(n):
                    qc.ry(params_ry[base + q], q)
                    qc.rz(params_rz[base + q], q)
        else:
            if parameter_sharing:
                for q in range(n):
                    if rotation == "ry":
                        qc.ry(params[layer_idx], q)
                    else:  # "rz"
                        qc.rz(params[layer_idx], q)
            else:
                base = layer_idx * n
                for q in range(n):
                    if rotation == "ry":
                        qc.ry(params[base + q], q)
                    else:  # "rz"
                        qc.rz(params[base + q], q)

    entanglement_pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        for _ in range(entanglement_depth):
            for (i, j) in entanglement_pairs:
                entangler_gate(qc, i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    # Attach metadata
    if rotation == "ryrz":
        qc.input_params = [params_ry, params_rz]  # type: ignore[attr-defined]
    else:
        qc.input_params = [params]  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    qc.rotation_type = rotation  # type: ignore[attr-defined]
    return qc


class ExtendedRealAmplitudes(QuantumCircuit):
    """Convenience subclass mirroring Qiskit's RealAmplitudes but with extended features."""

    def __init__(
        self,
        *,
        num_qubits: int,
        reps: int = 1,
        entanglement: Union[
            str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]
        ] = "full",
        entangler: str = "cx",
        rotation: str = "ry",
        entanglement_depth: int = 1,
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_sharing: bool = False,
        parameter_prefix: str = "theta",
        name: str = "ExtendedRealAmplitudes",
    ) -> None:
        built = extended_real_amplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            entangler=entangler,
            rotation=rotation,
            entanglement_depth=entanglement_depth,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_sharing=parameter_sharing,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.rotation_type = built.rotation_type  # type: ignore[attr-defined]


__all__ = ["ExtendedRealAmplitudes", "extended_real_amplitudes"]
