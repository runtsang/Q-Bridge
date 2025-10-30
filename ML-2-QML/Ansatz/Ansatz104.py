"""Extended RealAmplitudes ansatz builder (RY + optional RZ + CX + optional RZZ layers)."""
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


def extended_real_amplitudes(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    use_rzz: bool = False,
    use_rzx: bool = False,
    rzz_parameter_prefix: str = "phi",
    rzx_parameter_prefix: str = "psi",
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended RealAmplitudes ``QuantumCircuit``.

    The ansatz retains the original RY + CX structure but optionally adds
    RZ rotations on each qubit and RZZ entangling rotations between
    the same pairs that are used for the CX gates.  The additional
    layers are fully parameterised and can be enabled or disabled
    independently.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of rotation/entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement schedule for the CX and optional RZZ layers.
    skip_final_rotation_layer : bool, default False
        If True, the final layer of RY (and optional RZ) gates is omitted.
    insert_barriers : bool, default False
        If True, insert a barrier after each sub‑layer for visual clarity.
    parameter_prefix : str, default "theta"
        Prefix for the RY rotation parameters.
    use_rzz : bool, default False
        Whether to insert an RZZ rotation between each entangled pair.
    use_rzx : bool, default False
        Whether to insert an RZ rotation on each qubit before the CX gates.
    rzz_parameter_prefix : str, default "phi"
        Prefix for the RZZ rotation parameters.
    rzx_parameter_prefix : str, default "psi"
        Prefix for the RZ rotation parameters.
    name : str | None, default None
        Name of the resulting circuit.

    Returns
    -------
    QuantumCircuit
        The constructed parameterised ansatz.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "ExtendedRealAmplitudes")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    ry_params = ParameterVector(parameter_prefix, num_rot_layers * n)

    rz_params = None
    if use_rzx:
        rz_params = ParameterVector(rzx_parameter_prefix, num_rot_layers * n)

    pairs = _resolve_entanglement(n, entanglement)
    rzz_params = None
    if use_rzz and pairs:
        rzz_params = ParameterVector(rzz_parameter_prefix, num_rot_layers * len(pairs))

    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * n
        for q in range(n):
            qc.ry(ry_params[base + q], q)
            if use_rzx:
                qc.rz(rz_params[base + q], q)

    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if use_rzz:
            base_rzz = r * len(pairs)
            for idx, (i, j) in enumerate(pairs):
                qc.rzz(rzz_params[base_rzz + idx], i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)
        if insert_barriers:
            qc.barrier()

    # expose parameters for binding
    qc.input_params = {"ry": ry_params}
    if rz_params is not None:
        qc.input_params["rz"] = rz_params
    if rzz_params is not None:
        qc.input_params["rzz"] = rzz_params
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class ExtendedRealAmplitudes(QuantumCircuit):
    """Class‑style wrapper for the extended RealAmplitudes ansatz.

    The wrapper behaves like a normal ``QuantumCircuit`` but exposes
    the underlying parameter vectors via ``input_params`` and the
    number of rotation layers via ``num_rot_layers``.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        use_rzz: bool = False,
        use_rzx: bool = False,
        rzz_parameter_prefix: str = "phi",
        rzx_parameter_prefix: str = "psi",
        name: str = "ExtendedRealAmplitudes",
    ) -> None:
        built = extended_real_amplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            use_rzz=use_rzz,
            use_rzx=use_rzx,
            rzz_parameter_prefix=rzz_parameter_prefix,
            rzx_parameter_prefix=rzx_parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["ExtendedRealAmplitudes", "extended_real_amplitudes"]
