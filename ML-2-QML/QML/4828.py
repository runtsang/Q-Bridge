from __future__ import annotations

import numpy as np
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# ---------- Quantum Autoencoder Circuit ----------
def _quantum_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Variational autoencoder with a swap‑test based disentanglement."""
    qreg = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    creg = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qreg, creg)

    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.append(ansatz, list(range(num_latent + num_trash)))

    qc.barrier()
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_trash + num_latent + i)

    qc.h(aux)
    qc.measure(aux, creg[0])
    return qc


def QuantumAutoencoder(
    num_latent: int = 3,
    num_trash: int = 2,
) -> SamplerQNN:
    """Return a SamplerQNN that implements the variational autoencoder."""
    algorithm_globals.random_seed = 42
    sampler = StatevectorSampler()
    circuit = _quantum_autoencoder_circuit(num_latent, num_trash)
    return SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )


# ---------- Quantum Estimator ----------
def QuantumEstimator() -> EstimatorQNN:
    """Simple parameterised circuit that estimates the expectation of Y."""
    inp = Parameter("x1")
    wgt = Parameter("w1")
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(inp, 0)
    qc.rx(wgt, 0)

    obs = SparsePauliOp.from_list([("Y", 1)])
    estimator = StatevectorEstimator()
    return EstimatorQNN(
        circuit=qc,
        observables=obs,
        input_params=[inp],
        weight_params=[wgt],
        estimator=estimator,
    )


# ---------- Quantum Fully‑Connected Model ----------
class QuantumFCModel(tq.QuantumModule):
    """Quantum fully‑connected architecture inspired by Quantum‑NAT."""
    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["QuantumAutoencoder", "QuantumEstimator", "QuantumFCModel"]
