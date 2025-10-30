import torch
import torchquantum as tq
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.utils import algorithm_globals
import numpy as np

class QuanvolutionAutoencoder(tq.QuantumModule):
    """
    Quantum hybrid model combining a torchquantum quanvolution filter and a qiskit
    quantum auto‑encoder. The filter maps 2×2 image patches to 4‑dimensional
    quantum feature vectors; the auto‑encoder reconstructs the flattened
    feature vector using a RealAmplitudes ansatz and a swap‑test based
    measurement.
    """
    def __init__(self,
                 in_channels: int = 1,
                 conv_out_channels: int = 4,
                 conv_kernel: int = 2,
                 conv_stride: int = 2,
                 num_latent: int = 3,
                 num_trash: int = 2,
                 image_size: int = 28,
                 device: str | None = None):
        super().__init__()
        self.device = device or "cpu"
        self.n_wires = conv_out_channels * (image_size // conv_stride) ** 2
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.sampler = StatevectorSampler()
        algorithm_globals.random_seed = 42

    def _build_autoencoder_circuit(self, latent_params: np.ndarray) -> QuantumCircuit:
        """Constructs a swap‑test based auto‑encoder for a single sample."""
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        for i, val in enumerate(latent_params):
            qc.ry(val, i)
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=5)
        qc.append(ansatz, range(self.num_latent + self.num_trash))
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        flat_features = torch.cat(patches, dim=1)
        recon_batch = []
        for i in range(flat_features.shape[0]):
            params = flat_features[i].detach().cpu().numpy()
            if len(params) < self.num_latent:
                params = np.pad(params, (0, self.num_latent - len(params)), "constant")
            else:
                params = params[:self.num_latent]
            qc = self._build_autoencoder_circuit(params)
            result = self.sampler.run(qc).result()
            state = result.get_state()
            probs = np.abs(state) ** 2
            recon_vec = torch.from_numpy(probs[:784]).float().to(device)
            recon_batch.append(recon_vec)
        recon_tensor = torch.stack(recon_batch, dim=0)
        return recon_tensor.view(-1, 1, 28, 28)

__all__ = ["QuanvolutionAutoencoder"]
