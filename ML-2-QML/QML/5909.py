"""Quantum implementation of the hybrid quanvolution‑sampler model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator

class QuanvolutionSamplerHybrid(nn.Module):
    """
    Quantum‑centric counterpart of the hybrid model. For each 2×2 image patch
    a 4‑qubit circuit encodes the pixel values, applies a random layer,
    and measures all qubits. A separate 2‑qubit sampler circuit processes
    the top‑left and bottom‑right pixels. The resulting probability
    vectors are concatenated, flattened over the 14×14 patches, and
    classified by a classical linear head.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Classical linear head (shared with the classical version)
        self.fc = nn.Linear(4 * 14 * 14 + 2 * 14 * 14, num_classes)

        # Quantum resources
        self.backend = AerSimulator()
        self.filter_circuit = self._build_filter_circuit()
        self.sampler_circuit = self._build_sampler_circuit()

    def _build_filter_circuit(self) -> QuantumCircuit:
        """Builds a 4‑qubit circuit that encodes a 2×2 patch."""
        qc = QuantumCircuit(4)
        # Parameter vector for pixel values
        pixels = ParameterVector('p', 4)
        # Encode via Ry gates
        for i, qubit in enumerate(range(4)):
            qc.ry(pixels[i], qubit)
        # Random layer of 8 two‑qubit rotations
        for _ in range(8):
            qc.cx(0, 1)
            qc.ry(pixels[0], 0)
        # Measure all qubits
        qc.measure_all()
        return qc

    def _build_sampler_circuit(self) -> QuantumCircuit:
        """Builds the sampler circuit from qiskit‑machine‑learning."""
        # Using the provided SamplerQNN wrapper
        inputs = ParameterVector('in', 2)
        weights = ParameterVector('w', 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        qc.measure_all()
        return qc

    def _evaluate_circuit(self, circuit: QuantumCircuit, parameters: dict) -> torch.Tensor:
        """Runs the circuit on the simulator and returns probability vector."""
        bound_qc = circuit.bind_parameters(parameters)
        job = execute(bound_qc, self.backend, shots=1024)
        result = job.result()
        probs = result.get_counts()
        # Convert counts to probability vector over 2^n outcomes
        n_qubits = circuit.num_qubits
        probs_vec = torch.zeros(2 ** n_qubits)
        for outcome, count in probs.items():
            idx = int(outcome[::-1], 2)  # Qiskit uses little‑endian ordering
            probs_vec[idx] = count / 1024
        return probs_vec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        batch_size = x.size(0)
        # Prepare patches
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (batch, 1, 14, 14, 2, 2)
        patches = patches.squeeze(1)  # (batch, 14, 14, 2, 2)

        # Filter features
        filter_feats = []
        for b in range(batch_size):
            for i in range(14):
                for j in range(14):
                    pixel_vals = patches[b, i, j].flatten().tolist()  # 4 values
                    params = {'p0': pixel_vals[0], 'p1': pixel_vals[1],
                              'p2': pixel_vals[2], 'p3': pixel_vals[3]}
                    probs = self._evaluate_circuit(self.filter_circuit, params)
                    filter_feats.append(probs)
        filter_feats = torch.stack(filter_feats).view(batch_size, -1)  # (batch, 4*14*14)

        # Sampler features (top‑left and bottom‑right)
        sampler_feats = []
        for b in range(batch_size):
            for i in range(14):
                for j in range(14):
                    tl = patches[b, i, j, 0, 0].item()
                    br = patches[b, i, j, 1, 1].item()
                    params = {'in0': tl, 'in1': br,
                              'w0': 0.0, 'w1': 0.0, 'w2': 0.0, 'w3': 0.0}
                    probs = self._evaluate_circuit(self.sampler_circuit, params)
                    sampler_feats.append(probs)
        sampler_feats = torch.stack(sampler_feats).view(batch_size, -1)  # (batch, 2*14*14)

        # Concatenate and classify
        features = torch.cat([filter_feats, sampler_feats], dim=1)
        logits = self.fc(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionSamplerHybrid"]
