"""ScaledRealAmplitudes: an extensible RealAmplitudes variant.

This module builds a quantum circuit that extends the classic RealAmplitudes
ansatz by optionally inserting a mid‑layer rotation between the RY rotation
layer and the CX entanglers.  The extension allows the user to increase the
expressivity per repetition while keeping the original interface largely
unchanged.

Key features
------------
- **Mid‑layer rotation** – choose between RY, RZ or none.
- **Custom entanglement schedule** – string spec or user supplied function.
- **Depth control** – number of repetitions (`reps`) and optional final
  rotation layer (`skip_final_rotation_layer`).
- **Barriers** – optional barrier insertion after each sub‑layer.
- **Parameter naming** – all rotation parameters share a single prefix,
  but are internally partitioned per layer.

The function `scaled_real_amplitudes` returns a ready‑to‑use
`QuantumCircuit`.  The class `ScaledRealAmplitudes` behaves like the
function but can be instantiated as a circuit object.
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


def scaled_real_amplitudes(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    mid_rotation_type: str = "none",
    name: str | None = None,
) -> QuantumCircuit:
    """Build a ScaledRealAmplitudes-style ``QuantumCircuit`` (RY + optional mid‑layer + CX layers).

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit. Must be >= 1.
    reps : int, default 1
        Number of repetition blocks. Each block consists of a rotation layer,
        an optional mid‑layer rotation, and an entangling layer.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the two‑qubit entangling pairs.  The same conventions
        as the original ansatz are supported.
    skip_final_rotation_layer : bool, default False
        If ``True``, the final rotation layer after the last repetition is omitted.
    insert_barriers : bool, default False
        If ``True``, a barrier is inserted after each sub‑layer for readability.
    parameter_prefix : str, default "theta"
        Prefix for all rotation parameters.  The function generates a
        :class:`~qiskit.circuit.ParameterVector` of appropriate size.
    mid_rotation_type : str, default "none"
        Optional mid‑layer rotation type.  Accepted values are ``"none"``,
        ``"ry"``, and ``"rz"``.
    name : str | None, default None
        Name for the resulting circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.  The circuit has attributes
        ``input_params`` (the :class:`ParameterVector`) and ``num_rot_layers``
        (the number of rotation layers including the optional final one).

    Raises
    ------
    ValueError
        If ``num_qubits`` < 1, ``reps`` < 1, or an invalid ``mid_rotation_type`` is supplied.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if mid_rotation_type not in {"none", "ry", "rz"}:
        raise ValueError(f"Unsupported mid_rotation_type: {mid_rotation_type!r}")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "ScaledRealAmplitudes")

    # Determine number of rotation layers (including optional final one)
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    # Total number of parameters
    total_params = num_rot_layers * n
    if mid_rotation_type!= "none":
        total_params += reps * n

    params = ParameterVector(parameter_prefix, total_params)

    def _rotation_layer(layer_idx: int, base_idx: int) -> None:
        """Apply an RY rotation layer."""
        for q in range(n):
            qc.ry(params[base_idx + q], q)

    def _mid_rotation_layer(layer_idx: int, base_idx: int) -> None:
        """Apply the optional mid‑layer rotation."""
        if mid_rotation_type == "ry":
            for q in range(n):
                qc.ry(params[base_idx + q], q)
        elif mid_rotation_type == "rz":
            for q in range(n):
                qc.rz(params[base_idx + q], q)
        # ``mid_rotation_type == "none"`` is never called.

    pairs = _resolve_entanglement(n, entanglement)

    rot_base = 0
    mid_base = num_rot_layers * n

    for r in range(reps):
        _rotation_layer(r, rot_base + r * n)
        if insert_barriers:
            qc.barrier()
        if mid_rotation_type!= "none":
            _mid_rotation_layer(r, mid_base + r * n)
            if insert_barriers:
                qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps, rot_base + reps * n)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class ScaledRealAmplitudes(QuantumCircuit):
    """Class‑style wrapper that behaves like Qiskit's ``ScaledRealAmplitudes``."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        mid_rotation_type: str = "none",
        name: str = "ScaledRealAmplitudes",
    ) -> None:
        built = scaled_real_amplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            mid_rotation_type=mid_rotation_type,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["ScaledRealAmplitudes", "scaled_real_amplitudes"]
