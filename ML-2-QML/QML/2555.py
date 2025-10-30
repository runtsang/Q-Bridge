"""Quantum counterpart of ConvAutoencoderFusion.

This module implements a quantum version of the hybrid model.  It first
creates a convolution‑like quantum filter that encodes a 2×2 patch into
a set of qubits.  The resulting measurement probabilities are then fed
into a variational autoencoder circuit that performs a swap‑test style
encoding.  The output is a single‑qubit measurement that can be used as
a classical proxy for the latent representation.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.random import random_circuit
from qiskit.providers.aer import AerSimulator

class QuantumConvFilter:
    """A 2×2 convolution‑like filter implemented as a quantum circuit."""
    def __init__(self, kernel_size: int = 2, threshold: float = 127,
                 shots: int = 100) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = AerSimulator()
        # Build a parameterised circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"θ{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> np.ndarray:
        """Execute the filter on a 2×2 patch and return the probability
        of measuring |1> for each qubit.
        """
        flat = data.reshape(self.n_qubits)
        param_binds = {}
        for i, val in enumerate(flat):
            param_binds[self.theta[i]] = np.pi if val > self.threshold else 0.0
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=[param_binds])
        result = job.result()
        counts = result.get_counts(self.circuit)
        total_counts = sum(counts.values())
        ones_counts = np.zeros(self.n_qubits)
        for bitstring, count in counts.items():
            for i, bit in enumerate(reversed(bitstring)):
                if bit == '1':
                    ones_counts[i] += count
        probs = ones_counts / total_counts
        return probs

class QuantumAutoencoder:
    """Variational autoencoder circuit that accepts a probability vector."""
    def __init__(self, num_latent: int = 3, num_trash: int = 2,
                 reps: int = 5) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.num_qubits = num_latent + 2 * num_trash + 1
        self.backend = AerSimulator()
        # Build the circuit
        self.circuit = QuantumCircuit(self.num_qubits, 1)
        # Ansatz on latent + trash qubits
        ansatz = RealAmplitudes(num_latent + num_trash, reps=reps)
        self.circuit.append(ansatz, list(range(num_latent + num_trash)))
        self.circuit.barrier()
        # Swap test
        aux = num_latent + 2 * num_trash
        self.circuit.h(aux)
        for i in range(num_trash):
            self.circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        self.circuit.h(aux)
        self.circuit.measure(aux, 0)

    def run(self, probs: np.ndarray) -> float:
        """Run the autoencoder circuit with the given input probabilities
        encoded as rotation angles on the latent qubits.
        """
        angles = 2 * np.pi * probs[:self.num_latent]
        param_binds = {p: angle for p, angle in zip(list(self.circuit.parameters)[:self.num_latent], angles)}
        job = execute(self.circuit, self.backend, shots=100,
                      parameter_binds=[param_binds])
        result = job.result()
        counts = result.get_counts()
        # Expectation value of the measured auxiliary qubit
        exp_val = 0.0
        for key, count in counts.items():
            exp_val += (1 if key == '1' else -1) * count
        exp_val /= 100
        return exp_val

class ConvAutoencoderFusion:
    """Quantum hybrid model that mirrors the classical ConvAutoencoderFusion."""
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 127,
                 shots: int = 100,
                 num_latent: int = 3,
                 num_trash: int = 2) -> None:
        self.conv_filter = QuantumConvFilter(kernel_size=kernel_size,
                                             threshold=threshold,
                                             shots=shots)
        self.autoencoder = QuantumAutoencoder(num_latent=num_latent,
                                              num_trash=num_trash)

    def run(self, data: np.ndarray) -> float:
        """Process a 2×2 patch and return the autoencoder output."""
        probs = self.conv_filter.run(data)
        return self.autoencoder.run(probs)

__all__ = ["ConvAutoencoderFusion"]
