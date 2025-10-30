"""Hybrid quantum‑classical model combining torchquantum and a Qiskit
parameterised circuit.

The model encodes the input through a 4‑wire general encoder,
passes the state through a random‑layer based quantum module,
and then measures Pauli‑Z on all wires.  In addition a
separate Qiskit FCL circuit is executed on a subset of the
encoded features; its expectation value is concatenated with
the torchquantum output.  The final vector is normalised
with a 1‑D batch‑norm layer.

This demonstrates how a purely quantum sub‑routine can be
wrapped inside a larger quantum‑classical pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit

# --------------------------------------------------------------------------- #
#  Qiskit‑based fully connected layer
# --------------------------------------------------------------------------- #
class _QiskitFCL:
    """Parameterised quantum circuit that returns a single expectation value.

    The circuit is identical to the one in the seed but wrapped in a
    lightweight class that can be called from the forward pass.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 100) -> None:
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Evaluate the circuit for each theta in *thetas*.

        Parameters
        ----------
        thetas : array‑like
            One‑dimensional array of parameter values.

        Returns
        -------
        np.ndarray
            Expectation value for each input theta.
        """
        jobs = []
        for theta in thetas:
            bound = {self.theta: theta}
            job = qiskit.execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[bound],
            )
            jobs.append(job)

        expectations = []
        for job in jobs:
            result = job.result()
            counts = result.get_counts(self.circuit)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array(list(counts.keys())).astype(float)
            expectations.append(np.sum(states * probs))

        return np.array(expectations)


# --------------------------------------------------------------------------- #
#  Hybrid quantum module
# --------------------------------------------------------------------------- #
class QFCModel(tq.QuantumModule):
    """Hybrid quantum‑classical model inspired by the Quantum‑NAT paper.

    The model uses torchquantum for the main encoder and quantum
    layer, and a separate Qiskit FCL circuit to provide an
    additional scalar feature.  The two feature vectors are
    concatenated before normalisation.
    """

    class QLayer(tq.QuantumModule):
        """Random‑layer based quantum block with a few trainable gates."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )
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
        self.norm = nn.BatchNorm1d(self.n_wires + 1)  # +1 from Qiskit FCL

        # Initialise the Qiskit FCL circuit (single qubit, 100 shots)
        self.qiskit_fcl = _QiskitFCL(n_qubits=1, shots=100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)

        # Encode the classical features into the quantum state
        self.encoder(qdev, pooled)

        # Apply the trainable quantum block
        self.q_layer(qdev)

        # Measure all qubits → (bsz, 4)
        out_qt = self.measure(qdev)

        # Run the Qiskit FCL on the *first* pooled feature per sample
        # to obtain a single scalar per batch element.
        thetas = pooled[:, 0].cpu().numpy()
        fcl_out = self.qiskit_fcl.run(thetas)  # shape (bsz,)
        fcl_out = torch.from_numpy(fcl_out).to(x.device).unsqueeze(1)  # (bsz, 1)

        # Concatenate and normalise
        out = torch.cat([out_qt, fcl_out], dim=1)  # (bsz, 5)
        return self.norm(out)


__all__ = ["QFCModel"]
