import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.random import random_circuit
from qiskit.primitives import Sampler

class ConvAutoencoder:
    """
    Quantum counterpart that first applies a quanvolution circuit to each image patch
    and then compresses/reconstructs it with a quantum autoencoder.
    """
    def __init__(self,
                 conv_kernel: int = 2,
                 conv_threshold: float = 127,
                 latent_dim: int = 3,
                 trash: int = 2,
                 shots: int = 100) -> None:
        self.conv_kernel = conv_kernel
        self.conv_threshold = conv_threshold
        self.latent_dim = latent_dim
        self.trash = trash
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        self.quanv = self._build_quanv()
        self.autoenc = self._build_autoenc()

    def _build_quanv(self) -> QuantumCircuit:
        n = self.conv_kernel ** 2
        qc = QuantumCircuit(n)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n)]
        for i in range(n):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(n, 2)
        qc.measure_all()
        return qc

    def _build_autoenc(self) -> QuantumCircuit:
        num_latent = self.latent_dim
        num_trash = self.trash
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.compose(RealAmplitudes(num_latent + num_trash, reps=5),
                   range(0, num_latent + num_trash),
                   inplace=True)
        qc.barrier()
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        data : np.ndarray
            Grayscale image of shape (H, W).

        Returns
        -------
        np.ndarray
            Reconstructed image of the same shape.
        """
        H, W = data.shape
        n = self.conv_kernel ** 2

        # Extract patches
        patches = []
        for i in range(0, H - self.conv_kernel + 1):
            for j in range(0, W - self.conv_kernel + 1):
                patch = data[i:i + self.conv_kernel,
                             j:j + self.conv_kernel].flatten()
                patches.append(patch)
        patches = np.array(patches)  # (L, n)

        # Quanvolution: bind pixel values to rotation angles
        param_binds = []
        for p in patches:
            bind = {}
            for idx, val in enumerate(p):
                bind[self.quanv.parameters[idx]] = np.pi if val > self.conv_threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(self.quanv,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self.quanv)

        # Convert measurement counts to average |1> probabilities per qubit
        probs = []
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            probs.append(ones * val / (self.shots * n))
        probs = np.array(probs)  # (L,)

        # Quantum autoencoder: use probabilities as input parameters
        # For simplicity, bind the first len(self.autoenc.parameters) probabilities
        bind = {}
        for idx, val in enumerate(probs[:len(self.autoenc.parameters)]):
            bind[self.autoenc.parameters[idx]] = val
        job2 = qiskit.execute(self.autoenc,
                              self.backend,
                              shots=self.shots,
                              parameter_binds=[bind])
        result2 = job2.result().get_counts(self.autoenc)

        # Interpret counts as reconstructed probabilities
        recon = []
        for key, val in result2.items():
            ones = sum(int(bit) for bit in key)
            recon.append(ones * val / (self.shots * (self.latent_dim + 2 * self.trash + 1)))
        recon = np.array(recon).reshape(-1, self.conv_kernel, self.conv_kernel)

        # Reconstruct the image by averaging overlapping patches
        recon_img = np.zeros((H, W))
        count = np.zeros((H, W))
        idx = 0
        for i in range(0, H - self.conv_kernel + 1):
            for j in range(0, W - self.conv_kernel + 1):
                recon_img[i:i + self.conv_kernel,
                          j:j + self.conv_kernel] += recon[idx]
                count[i:i + self.conv_kernel,
                      j:j + self.conv_kernel] += 1
                idx += 1
        return recon_img / count

__all__ = ["ConvAutoencoder"]
