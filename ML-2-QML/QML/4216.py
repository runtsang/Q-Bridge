from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA


# ----------------------------------------------------------------------
# Photonic‑style fraud layer parameters (used only for documentation)
# ----------------------------------------------------------------------
class FraudLayerParameters:
    def __init__(
        self,
        bs_theta: float,
        bs_phi: float,
        phases: Tuple[float, float],
        squeeze_r: Tuple[float, float],
        squeeze_phi: Tuple[float, float],
        displacement_r: Tuple[float, float],
        displacement_phi: Tuple[float, float],
        kerr: Tuple[float, float],
    ):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr


# ----------------------------------------------------------------------
# Quantum self‑attention block
# ----------------------------------------------------------------------
class QuantumSelfAttention:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


# ----------------------------------------------------------------------
# Hybrid quantum autoencoder
# ----------------------------------------------------------------------
class HybridAutoencoder:
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        fraud_layers: Iterable[FraudLayerParameters] | None = None,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.fraud_layers = list(fraud_layers or [])

        # --- Encoding ansatz ------------------------------------------------
        self.encoder_ansatz = RealAmplitudes(input_dim, reps=3)

        # --- Attention block ------------------------------------------------
        self.attention = QuantumSelfAttention(n_qubits=input_dim)

        # --- Fraud‑layer circuit --------------------------------------------
        self.fraud_circuit = QuantumCircuit()
        for layer in self.fraud_layers:
            self._add_fraud_layer(layer)

        # --- Full circuit ---------------------------------------------------
        self.circuit = QuantumCircuit()
        self.circuit.compose(self.encoder_ansatz, inplace=True)
        # dummy params for attention; will be set at runtime
        self.circuit.compose(
            self.attention._build_circuit(
                np.zeros(3 * input_dim), np.zeros(input_dim - 1)
            ),
            inplace=True,
        )
        self.circuit.compose(self.fraud_circuit, inplace=True)

        # Sampler and QNN wrapper
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.encoder_ansatz.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _add_fraud_layer(self, params: FraudLayerParameters) -> None:
        # Simple mapping of photonic gates to standard rotations
        self.fraud_circuit.rx(params.bs_theta, 0)
        self.fraud_circuit.ry(params.bs_phi, 0)
        for i, phase in enumerate(params.phases):
            self.fraud_circuit.rz(phase, 0)
        for r, phi in zip(params.squeeze_r, params.squeeze_phi):
            self.fraud_circuit.rx(r, 0)
            self.fraud_circuit.ry(phi, 0)
        for r, phi in zip(params.displacement_r, params.displacement_phi):
            self.fraud_circuit.rz(r, 0)
            self.fraud_circuit.rx(phi, 0)
        for k in params.kerr:
            self.fraud_circuit.rz(k, 0)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        # For simplicity we ignore actual data encoding; only weight params are learned
        return self.qnn.forward(inputs)

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 10,
        shots: int = 1024,
        verbose: bool = False,
    ) -> list[float]:
        """Very light‑weight COBYLA training over encoder weights."""
        opt = COBYLA(maxiter=epochs)
        history: list[float] = []

        def objective(weights: np.ndarray) -> float:
            # load weights into ansatz
            for param, val in zip(self.encoder_ansatz.parameters, weights):
                param.data = torch.tensor(val, dtype=torch.float32)
            preds = self.predict(data)
            mse = float(np.mean((preds - data) ** 2))
            if verbose:
                print(f"Epoch {len(history)} MSE {mse:.6f}")
            history.append(mse)
            return mse

        init_w = np.random.rand(len(self.encoder_ansatz.parameters))
        opt.optimize(init_w, objective)
        return history


# ----------------------------------------------------------------------
# Factory helper mirroring classical API
# ----------------------------------------------------------------------
def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    fraud_layers: Iterable[FraudLayerParameters] | None = None,
) -> HybridAutoencoder:
    return HybridAutoencoder(input_dim, latent_dim=latent_dim, fraud_layers=fraud_layers)


__all__ = [
    "HybridAutoencoder",
    "FraudLayerParameters",
    "QuantumSelfAttention",
]
