"""Quantum helper module that supplies a variational feature extractor, an auto‑encoder circuit, and a quantum LSTM.

All functions are pure quantum and can be called from the classical module via a callback.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import RealAmplitudes
import torchquantum as tq
import torchquantum.functional as tqf
import qutip as qt
import networkx as nx


# --------------------------------------------------------------------------- #
# Variational classifier circuit – from reference 1
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector, List[SparsePauliOp]]:
    """
    Construct a layered variational circuit with data encoding and a trainable ansatz.

    Returns
    -------
    circuit : QuantumCircuit
        The variational circuit with symbolic parameters.
    encoding : ParameterVector
        The data‑encoding parameters.
    weights : ParameterVector
        The variational parameters.
    observables : list[SparsePauliOp]
        One Z observable per qubit for expectation extraction.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        # entangling layer
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, encoding, weights, observables


def quantum_feature_extractor(
    x: torch.Tensor,
    circuit: QuantumCircuit,
    observables: List[SparsePauliOp],
) -> torch.Tensor:
    """
    Evaluate the circuit for each sample in ``x`` and return the expectation values.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(batch, num_qubits)`` – data to encode.
    circuit : QuantumCircuit
        The variational circuit with symbolic parameters.
    observables : list[SparsePauliOp]
        Observables to measure.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, num_qubits)`` – expectation values for every qubit.
    """
    batch = x.shape[0]
    num_qubits = circuit.num_qubits
    features: List[List[float]] = []

    for i in range(batch):
        params = dict(zip(circuit.parameters, x[i].tolist()))
        state = Statevector(circuit.bind_parameters(params))
        exp = [state.expectation_value(obs).real for obs in observables]
        features.append(exp)

    return torch.tensor(features, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Quantum auto‑encoder circuit – from reference 2
# --------------------------------------------------------------------------- #
def autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """
    Build a small quantum auto‑encoder that uses a swap‑test to recover the latent state.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Ansatz – a few repetitions of RealAmplitudes
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.append(ansatz, list(range(num_latent + num_trash)))

    qc.barrier()

    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc


# --------------------------------------------------------------------------- #
# Quantum LSTM – from reference 3
# --------------------------------------------------------------------------- #
class QLSTMQuantum(tq.QuantumModule):
    """
    A quantum‑enhanced LSTM cell where each gate is a small quantum circuit.
    Implementation uses torchquantum for differentiable simulation.
    """

    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs: List[torch.Tensor] = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# --------------------------------------------------------------------------- #
# Fidelity‑graph utilities – from reference 4 (quantum version)
# --------------------------------------------------------------------------- #
def fidelity_graph_qt(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """
    Build a weighted adjacency graph from the fidelities of pure quantum states.

    Parameters
    ----------
    states : list[qt.Qobj]
        List of pure states.
    threshold : float
        Minimum fidelity for a unit‑weight edge.
    secondary : float | None, optional
        Secondary threshold for a lighter edge.
    secondary_weight : float, default 0.5
        Weight assigned to secondary edges.

    Returns
    -------
    nx.Graph
        Weighted graph where nodes represent states.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j, b in enumerate(states[i + 1 :], i + 1):
            fid = abs((a.dag() * b)[0, 0]) ** 2
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
    return G
