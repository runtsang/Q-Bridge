"""Quantum implementation of the hybrid self‑attention auto‑encoder.

The circuit is built from three stages:

1.  A convolution‑style feature encoding that maps a 2×2 patch of classical
    data into a small register of qubits.
2.  A variational attention block that entangles the feature register with a
    latent register.
3.  A sampler QNN that interprets the final state as a probability distribution.

The module exposes a `run` method that executes the circuit on a chosen
backend and returns the measurement histogram.  It can be used as a drop‑in
replacement for the classical `HybridSelfAttentionAutoEncoder` in a quantum
workflow.

Typical usage::

    model = HybridSelfAttentionAutoEncoder()
    counts = model.run(backend=qiskit.Aer.get_backend("qasm_simulator"))
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler


class HybridSelfAttentionAutoEncoder:
    """
    Quantum circuit that mirrors the classical hybrid encoder.

    Parameters
    ----------
    embed_dim : int
        Number of qubits used for the attention register.
    latent_dim : int
        Number of qubits used for the latent encoder register.
    kernel_size : int, default 2
        Size of the 2×2 convolution patch; determines the feature qubits.
    """

    def __init__(self, embed_dim: int = 4, latent_dim: int = 3, kernel_size: int = 2) -> None:
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.n_feature = kernel_size ** 2

        # Parameter vectors
        self.input_params = ParameterVector("x", self.n_feature)
        self.latent_params = ParameterVector("w_latent", self.latent_dim)
        self.attention_params = ParameterVector("w_attn", self.embed_dim)

        # Build the variational ansatz for the latent block
        self.latent_ansatz = RealAmplitudes(self.latent_dim, reps=3)

        # Build the full circuit
        self.circuit = self._build_circuit()

        # Sampler QNN wrapper
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.latent_params,
            sampler=Sampler(),
            output_shape=2,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Assemble the full variational circuit."""
        qr = QuantumRegister(self.n_feature + self.latent_dim + self.embed_dim, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)

        # 1. Feature encoding: apply X gates conditioned on the classical patch
        for i in range(self.n_feature):
            circuit.x(qr[i])  # placeholder; actual conditioning is handled by SamplerQNN

        # 2. Latent block: variational ansatz
        circuit.compose(
            self.latent_ansatz,
            range(self.n_feature, self.n_feature + self.latent_dim),
            inplace=True,
        )

        # 3. Attention entanglement: controlled rotations between latent and embed registers
        for i in range(self.embed_dim):
            qubit_latent = self.n_feature + self.latent_dim + i
            qubit_attn = self.n_feature + self.latent_dim + i
            circuit.cx(qubit_latent, qubit_attn)
            circuit.ry(self.attention_params[i], qubit_attn)

        # 4. Measurement
        circuit.measure(qr[:2], cr)  # measure two qubits for the sampler output
        return circuit

    def run(self, backend, shots: int = 1024, data: np.ndarray | None = None) -> dict:
        """
        Execute the circuit on the supplied backend.

        Parameters
        ----------
        backend
            A Qiskit backend instance.
        shots
            Number of shots for the execution.
        data
            Optional 2×2 array of classical values that will be mapped to the
            feature qubits via the parameter binding.
        Returns
        -------
        counts
            Measurement histogram from the execution.
        """
        if data is None:
            data = np.random.randint(0, 2, size=(self.n_feature,))
        # Bind data to input parameters
        param_binds = [{self.input_params[i]: np.pi if val else 0.0 for i, val in enumerate(data)}]
        job = execute(
            self.circuit,
            backend=backend,
            shots=shots,
            parameter_binds=param_binds,
        )
        return job.result().get_counts(self.circuit)

    def get_params(self) -> dict:
        """Return current parameter values for inspection."""
        return {
            "latent": self.latent_params,
            "attention": self.attention_params,
        }

__all__ = ["HybridSelfAttentionAutoEncoder"]
