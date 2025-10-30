"""Quantum‑centric counterpart of the hybrid architecture using TorchQuantum and Qiskit."""
from __future__ import annotations

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np

# Quantum modules -----------------------------------------------------------

class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random 2‑qubit quantum kernel to 2×2 patches of a 28×28 image."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        bsz = x.shape[0]
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(q_device, data)
                self.q_layer(q_device)
                measurement = self.measure(q_device)
                patches.append(measurement.view(bsz, 4))
        q_device.reset_states(bsz)
        q_device.states = torch.cat(patches, dim=1).view(-1, 1)


class QuantumKernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.QuantumModule(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


class QuantumAutoencoder(tq.QuantumModule):
    """Autoencoder built on a Qiskit SamplerQNN."""
    def __init__(self, num_latent: int = 3, num_trash: int = 2) -> None:
        super().__init__()
        from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
        from qiskit.circuit.library import RealAmplitudes
        from qiskit.primitives import StatevectorSampler as Sampler
        from qiskit_machine_learning.neural_networks import SamplerQNN

        def ansatz(num_qubits):
            return RealAmplitudes(num_qubits, reps=5)

        def auto_encoder_circuit(num_latent, num_trash):
            qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
            cr = ClassicalRegister(1, "c")
            circuit = QuantumCircuit(qr, cr)
            circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
            circuit.barrier()
            auxiliary_qubit = num_latent + 2 * num_trash
            # swap test
            circuit.h(auxiliary_qubit)
            for i in range(num_trash):
                circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
            circuit.h(auxiliary_qubit)
            circuit.measure(auxiliary_qubit, cr[0])
            return circuit

        self.circuit = auto_encoder_circuit(num_latent, num_trash)
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=Sampler(),
        )

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        # forward through the Qiskit SamplerQNN
        # The Qiskit SamplerQNN expects classical inputs; we use the device to send data.
        # For simplicity, we treat inputs as a placeholder and rely on the SamplerQNN to handle weights.
        # In practice, one would map x to the Qiskit parameters.
        pass  # Placeholder: full integration would require a custom bridge


class QuantumEstimator(tq.QuantumModule):
    """EstimatorQNN built on a Qiskit Estimator."""
    def __init__(self) -> None:
        super().__init__()
        from qiskit.circuit import Parameter
        from qiskit import QuantumCircuit
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit.primitives import StatevectorEstimator as Estimator

        params1 = [Parameter("input1"), Parameter("weight1")]
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc1.ry(params1[0], 0)
        qc1.rx(params1[1], 0)

        observable1 = qc1.to_matrix()  # simple placeholder

        self.estimator = Estimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=qc1,
            observables=observable1,
            input_params=[params1[0]],
            weight_params=[params1[1]],
            estimator=self.estimator,
        )

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        # placeholder: in practice, would map x to circuit parameters and run estimator
        return torch.zeros(x.shape[0], 1)


class QuanvolutionAutoencoderQNN(tq.QuantumModule):
    """
    Quantum‑centric composite model mirroring the classical implementation.
    All sub‑modules are quantum or quantum‑inspired, but the public API
    (fit_features, forward) is identical to the classical counterpart.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        latent_dim: int = 3,
        num_trash: int = 2,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.kernel = QuantumKernel()
        self.autoencoder = QuantumAutoencoder(num_latent=latent_dim, num_trash=num_trash)
        self.estimator = QuantumEstimator()
        self._training_features: torch.Tensor | None = None

    def fit_features(self, features: torch.Tensor) -> None:
        """Store a reference set of features for kernel evaluation."""
        self._training_features = features.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        q_device = tq.QuantumDevice(n_wires=self.qfilter.n_wires, bsz=x.shape[0])
        self.qfilter(q_device, x)
        qfeat = q_device.states.view(x.shape[0], -1)

        if self._training_features is not None:
            # compute kernel matrix between input and stored reference set
            kmat = torch.stack(
                [self.kernel(qfeat[i:i+1], self._training_features) for i in range(qfeat.shape[0])],
                dim=0,
            )
            qfeat = kmat

        # autoencoder and estimator are placeholders; in a full implementation they would
        # be invoked with quantum devices and appropriate parameter mappings.
        out = self.estimator(q_device, qfeat)
        return out


__all__ = ["QuanvolutionAutoencoderQNN"]
