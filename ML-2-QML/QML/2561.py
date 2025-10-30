"""Hybrid quantum autoencoder combining a RealAmplitudes encoder/decoder
with a self‑attention subcircuit.  The structure mirrors the classical
`HybridAutoencoder` but replaces linear layers with a parameterised
quantum circuit, enabling variational optimisation on a quantum device.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA

# Quantum self‑attention subcircuit ------------------------------------------
class QuantumSelfAttention:
    """Parameterised self‑attention block that can be inserted into any
    quantum circuit.  It emulates the behaviour of the reference
    SelfAttention.py but is expressed as a reusable circuit fragment.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def build(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

# Hybrid autoencoder ---------------------------------------------------------
class HybridQuantumAutoencoder:
    """Variational autoencoder that uses a RealAmplitudes ansatz for the
    encoder/decoder and a QuantumSelfAttention subcircuit in the middle.
    """
    def __init__(
        self,
        num_latent: int,
        num_trash: int,
        attention_qubits: int,
        reps: int = 2,
        backend: qiskit.providers.BaseBackend | None = None,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.attention_qubits = attention_qubits
        self.reps = reps
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

        # Parameter shapes
        self.encoder_params = np.random.rand(num_latent * 2 * reps)
        self.decoder_params = np.random.rand(num_latent * 2 * reps)
        self.attn_rot_params = np.random.rand(attention_qubits * 3)
        self.attn_ent_params = np.random.rand(attention_qubits - 1)

        # Build fixed parts of the circuit
        self.circuit = self._build_circuit()

        # Sampler QNN
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self._all_params(),
            interpret=self._interpret,
            output_shape=(num_latent,),
            sampler=self.sampler,
        )

    def _all_params(self) -> list[np.ndarray]:
        return [
            self.encoder_params,
            self.decoder_params,
            self.attn_rot_params,
            self.attn_ent_params,
        ]

    def _interpret(self, x: np.ndarray) -> np.ndarray:
        # Return the raw statevector amplitudes as the latent vector
        return x

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(
            self.num_latent + self.num_trash + self.attention_qubits,
            "q",
        )
        cr = ClassicalRegister(self.attention_qubits, "c")
        qc = QuantumCircuit(qr, cr)

        # Encoder
        encoder = RealAmplitudes(
            self.num_latent,
            reps=self.reps,
            insert_barriers=False,
        )
        qc.append(encoder, range(self.num_latent))

        # Attention subcircuit
        attn = QuantumSelfAttention(self.attention_qubits)
        attn_circ = attn.build(
            rotation_params=self.attn_rot_params,
            entangle_params=self.attn_ent_params,
        )
        qc.append(attn_circ, range(self.num_latent + self.num_trash,
                                   self.num_latent + self.num_trash + self.attention_qubits))

        # Decoder
        decoder = RealAmplitudes(
            self.num_latent,
            reps=self.reps,
            insert_barriers=False,
        )
        qc.append(decoder, range(self.num_latent))

        return qc

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Return the latent representation for the given inputs."""
        # Inputs are ignored in this toy example; the circuit is fully parameterised.
        return self.qnn(inputs)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Decode latent vectors back to the original dimension."""
        # For simplicity we reuse the same QNN; in practice a separate decoder
        # circuit would be optimised.
        return self.qnn(latents)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(inputs))

    def train(self, data: np.ndarray, *, epochs: int = 50, learning_rate: float = 1e-3) -> list[float]:
        """Variational training loop using COBYLA to minimise MSE."""
        history: list[float] = []
        opt = COBYLA(maxiter=2000, disp=False, tol=1e-5)

        def objective(params: np.ndarray) -> float:
            # Unpack params
            idx = 0
            self.encoder_params = params[idx : idx + len(self.encoder_params)]
            idx += len(self.encoder_params)
            self.decoder_params = params[idx : idx + len(self.decoder_params)]
            idx += len(self.decoder_params)
            self.attn_rot_params = params[idx : idx + len(self.attn_rot_params)]
            idx += len(self.attn_rot_params)
            self.attn_ent_params = params[idx : idx + len(self.attn_ent_params)]

            # Re‑build circuit with updated parameters
            self.circuit = self._build_circuit()
            self.qnn = SamplerQNN(
                circuit=self.circuit,
                input_params=[],
                weight_params=self._all_params(),
                interpret=self._interpret,
                output_shape=(self.num_latent,),
                sampler=self.sampler,
            )
            recon = self.forward(data)
            loss = np.mean((recon - data) ** 2)
            return loss

        # Flatten initial parameters
        init_params = np.concatenate(self._all_params())
        best_params = opt.minimize(objective, init_params)
        # Update model with best parameters
        idx = 0
        self.encoder_params = best_params[idx : idx + len(self.encoder_params)]
        idx += len(self.encoder_params)
        self.decoder_params = best_params[idx : idx + len(self.decoder_params)]
        idx += len(self.decoder_params)
        self.attn_rot_params = best_params[idx : idx + len(self.attn_rot_params)]
        idx += len(self.attn_rot_params)
        self.attn_ent_params = best_params[idx : idx + len(self.attn_ent_params)]
        # Rebuild final circuit
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self._all_params(),
            interpret=self._interpret,
            output_shape=(self.num_latent,),
            sampler=self.sampler,
        )
        return history

__all__ = [
    "HybridQuantumAutoencoder",
    "QuantumSelfAttention",
]
