"""Quantum kernel and autoencoder integration.

This module defines a quantum kernel using TorchQuantum and a quantum autoencoder
using Qiskit’s SamplerQNN.  The class exposes a common interface for kernel
computation and latent encoding.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class QuantumKernelAutoencoder:
    """Quantum kernel and autoencoder wrapper."""
    def __init__(self,
                 n_wires: int = 4,
                 latent_dim: int = 3,
                 trash_dim: int = 2,
                 ansatz_reps: int = 5) -> None:
        # Quantum kernel
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = self._build_ansatz(ansatz_reps)
        self.kernel = self._build_kernel()

        # Quantum autoencoder
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.autoencoder_circuit = self._build_autoencoder_circuit()
        self.qnn = self._build_qnn()

    # -------------------- Kernel --------------------
    def _build_ansatz(self, reps: int) -> tq.QuantumModule:
        """Construct a simple Ry ansatz with given repetitions."""
        class RyAnsatz(tq.QuantumModule):
            def __init__(self, n_wires: int, reps: int):
                super().__init__()
                self.n_wires = n_wires
                self.reps = reps

            @tq.static_support
            def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
                q_device.reset_states(x.shape[0])
                for _ in range(self.reps):
                    for w in range(self.n_wires):
                        params = x[:, w] if func_name_dict["ry"].num_params else None
                        func_name_dict["ry"](q_device, wires=[w], params=params)
        return RyAnsatz(self.n_wires, reps)

    def _build_kernel(self) -> tq.QuantumModule:
        """Wrap the ansatz to compute an overlap kernel."""
        class KernelModule(tq.QuantumModule):
            def __init__(self, ansatz: tq.QuantumModule):
                super().__init__()
                self.ansatz = ansatz
                self.q_device = tq.QuantumDevice(n_wires=ansatz.n_wires)

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                x = x.reshape(1, -1)
                y = y.reshape(1, -1)
                self.ansatz(self.q_device, x)
                self.ansatz(self.q_device, -y)
                return torch.abs(self.q_device.states.view(-1)[0])
        return KernelModule(self.ansatz)

    def compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute quantum kernel value."""
        return self.kernel(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two sets of samples."""
        return np.array([[self.compute_kernel(x, y).item() for y in b] for x in a])

    # -------------------- Autoencoder --------------------
    def _build_autoencoder_circuit(self) -> QuantumCircuit:
        """Construct a swap‑test based autoencoder circuit."""
        num_latent = self.latent_dim
        num_trash = self.trash_dim
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Feature encoding with RealAmplitudes
        circuit.compose(RealAmplitudes(num_latent + num_trash, reps=5),
                        range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()

        # Swap‑test
        aux = num_latent + 2 * num_trash
        circuit.h(aux)
        for i in range(num_trash):
            circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])
        return circuit

    def _build_qnn(self) -> SamplerQNN:
        """Wrap the autoencoder circuit as a SamplerQNN."""
        sampler = StatevectorSampler()
        # Interpret function: return measurement probability as latent vector
        def interpret(x: np.ndarray) -> np.ndarray:
            return x
        return SamplerQNN(
            circuit=self.autoencoder_circuit,
            input_params=[],
            weight_params=self.autoencoder_circuit.parameters,
            interpret=interpret,
            output_shape=(self.latent_dim,),
            sampler=sampler,
        )

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Encode classical data into latent representation using the quantum autoencoder."""
        # Placeholder: returns zeros
        return np.zeros((inputs.shape[0], self.latent_dim))

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Placeholder decode: reconstructs input from latent vector."""
        return np.zeros((latents.shape[0], self.latent_dim))

__all__ = ["QuantumKernelAutoencoder"]
