"""Hybrid quantum model that consumes the 4‑dim latent vector from the classical encoder,
applies a variational circuit, and uses a Qiskit sampler for probabilistic output.

The quantum module is built on torchquantum for the main circuit and on Qiskit for the
SamplerQNN, demonstrating a combination of simulation back‑ends.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN


class HybridQuantumNAT(tq.QuantumModule):
    """Quantum counterpart to the classical HybridQuantumNAT.

    The forward pass:
    1. Encodes a 4‑dim latent vector into a 4‑wire quantum device.
    2. Applies a parametric variational layer (RandomLayer + RX/RY/RZ/CRX).
    3. Measures all qubits (Pauli‑Z) and normalises to a probability vector.
    4. Runs a Qiskit SamplerQNN on the same latent vector to produce an alternative
       probability distribution, illustrating hybrid back‑ends.
    """

    class VariationalLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[0, 3])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[2, 1], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, latent_dim: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_wires = 4
        # Encoder that maps the latent vector to the quantum device
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Variational layer
        self.var_layer = self.VariationalLayer()
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Normalisation
        self.norm = nn.BatchNorm1d(self.n_wires)

        # Qiskit SamplerQNN for cross‑validation
        self._qiskit_sampler = self._build_qiskit_sampler()

    def _build_qiskit_sampler(self) -> QiskitSamplerQNN:
        """Construct a Qiskit SamplerQNN that mirrors the parameterised circuit."""
        # Parameter vector for inputs (latent)
        inputs = ParameterVector("latent", self.latent_dim)
        # Parameter vector for weights
        weights = ParameterVector("weight", 4)

        qc = QuantumCircuit(self.latent_dim)
        # Simple 2‑qubit sampler pattern (expandable to 4 qubits)
        for i in range(self.latent_dim):
            qc.ry(inputs[i], i)
        qc.cx(0, 1)
        for i in range(self.latent_dim):
            qc.ry(weights[i], i)

        # Wrap into a SamplerQNN
        sampler = StatevectorSampler()
        return QiskitSamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler,
        )

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return quantum measurement probabilities and Qiskit sampler probabilities.

        Parameters
        ----------
        latent : torch.Tensor
            Batch of latent vectors (batch_size, 4) produced by the classical encoder.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            * quantum_probs: batch_size × 4 probability vector from the variational circuit.
            * sampler_probs: batch_size × 4 probability vector from the Qiskit SamplerQNN.
        """
        bsz = latent.size(0)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=latent.device, record_op=True)

        # Encode the latent vector into the quantum device
        self.encoder(qdev, latent)

        # Variational circuit
        self.var_layer(qdev)

        # Measurement and normalisation
        raw = self.measure(qdev)  # batch_size × 4
        quantum_probs = F.softmax(self.norm(raw), dim=-1)

        # Qiskit sampler (performed sample‑by‑sample due to state‑vector nature)
        sampler_probs_list = []
        for idx in range(bsz):
            # Build a copy of the circuit with the specific latent values
            param_dict = {f"latent_{i}": float(latent[idx, i].item()) for i in range(self.latent_dim)}
            # Evaluate sampler
            res = self._qiskit_sampler.sample(param_dict, shots=1024)
            probs = res.sample_counts
            # Convert counts to probability vector over 4 basis states
            probs_vec = torch.tensor(
                [probs.get(f"{i:04b}", 0) / 1024 for i in range(16)],
                dtype=torch.float32,
                device=latent.device,
            )
            # Keep only the first 4 probabilities for consistency
            sampler_probs_list.append(probs_vec[:self.n_wires])

        sampler_probs = torch.stack(sampler_probs_list, dim=0)
        sampler_probs = F.softmax(sampler_probs, dim=-1)

        return quantum_probs, sampler_probs


__all__ = ["HybridQuantumNAT"]
