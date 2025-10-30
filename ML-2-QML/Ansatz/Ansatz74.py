"""RealAmplitudes variant with alternating RY/RX rotation layers and optional parameter sharing and entanglement ordering."""
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


def real_amplitudes_alternating_controlled(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    shared_params: bool = False,
    reverse_entanglement: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a RealAmplitudes ansatz with alternating RY/RX rotation layers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default=1
        Number of repetition blocks. Each block contains one rotation layer
        followed by an entanglement layer. If ``skip_final_rotation_layer`` is
        ``False`` an extra rotation layer is appended after the last block.
    entanglement : str or sequence or callable, default="full"
        Specification of which qubit pairs to entangle. Accepted strings are
        ``"full"``, ``"linear"``, and ``"circular"``. A sequence of pairs or
        a callable that returns such a sequence is also accepted.
    skip_final_rotation_layer : bool, default=False
        If ``True`` the final rotation layer after the last repetition is omitted.
    insert_barriers : bool, default=False
        If ``True`` a barrier is inserted before and after each entanglement
        layer for better readability in circuit visualisations.
    parameter_prefix : str, default="theta"
        Prefix used for the parameters in the circuit.
    shared_params : bool, default=False
        When ``True`` all qubits in a given rotation layer share the same
        rotation angle. This reduces the number of parameters from ``n * L``
        to ``L`` where ``L`` is the number of rotation layers.
    reverse_entanglement : bool, default=False
        If ``True`` the entanglement pairs are applied in reverse order for
        each repetition, providing a simple symmetry variation.
    name : str, optional
        Name of the resulting circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit. The circuit has attributes
        ``input_params`` (the :class:`~qiskit.circuit.ParameterVector` used)
        and ``num_rot_layers`` (the number of rotation layers).

    Notes
    -----
    - The rotation layers alternate between :func:`~qiskit.circuit.QuantumCircuit.ry`
      and :func:`~qiskit.circuit.QuantumCircuit.rx` depending on the layer index.
    - The entanglement layer is applied after each rotation layer.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingControlled")

    # Determine number of rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    # Parameter vector length depends on whether parameters are shared
    param_len = num_rot_layers if shared_params else num_rot_layers * n
    params = ParameterVector(parameter_prefix, param_len)

    def _rot(layer: int) -> None:
        """Apply the rotation layer for the given ``layer`` index."""
        if shared_params:
            idx = layer
            angle = params[idx]
            if layer % 2 == 0:
                for q in range(n):
                    qc.ry(angle, q)
            else:
                for q in range(n):
                    qc.rx(angle, q)
        else:
            base = layer * n
            if layer % 2 == 0:
                for q in range(n):
                    qc.ry(params[base + q], q)
            else:
                for q in range(n):
                    qc.rx(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)
    if reverse_entanglement:
        pairs = list(reversed(pairs))

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

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingControlled(QuantumCircuit):
    """Convenience wrapper for the controlled‑variation RealAmplitudes ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        shared_params: bool = False,
        reverse_entanglement: bool = False,
        name: str = "RealAmplitudesAlternatingControlled",
    ) -> None:
        built = real_amplitudes_alternating_controlled(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            shared_params,
            reverse_entanglement,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = [
    "RealAmplitudesAlternatingControlled",
    "real_amplitudes_alternating_controlled",
]
