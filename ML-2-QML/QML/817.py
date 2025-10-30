import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit import Aer, execute

class Conv:
    """
    Quantum convolutional filter that applies a variational circuit to each
    kernel‑sized patch of an image.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel (e.g., 2 for a 2×2 patch).
    backend : qiskit.providers.Backend, optional
        Quantum backend to run the circuits on.  Defaults to the Aer qasm
        simulator.
    shots : int, optional
        Number of shots per circuit execution.
    threshold : float, optional
        Pixel intensity threshold used to set the rotation angles.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 1024,
        threshold: float = 0.5,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots

        # Build a reusable variational block
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"θ{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        # Add a simple entangling layer
        for i in range(self.n_qubits - 1):
            self.circuit.cz(i, i + 1)
        self.circuit.barrier()
        self.circuit.measure_all()

        self.backend = backend or Aer.get_backend("qasm_simulator")

    def _patches_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        Extract non‑overlapping patches of size (kernel_size, kernel_size)
        from a 2‑D image.

        Parameters
        ----------
        image : np.ndarray
            2‑D array of shape (H, W).

        Returns
        -------
        np.ndarray
            Array of shape (num_patches, n_qubits) where each row is a flattened
            patch.
        """
        H, W = image.shape
        assert H % self.kernel_size == 0 and W % self.kernel_size == 0, (
            "Image dimensions must be divisible by kernel_size."
        )
        patches = []
        for i in range(0, H, self.kernel_size):
            for j in range(0, W, self.kernel_size):
                patch = image[i : i + self.kernel_size, j : j + self.kernel_size]
                patches.append(patch.flatten())
        return np.array(patches)

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum filter on a single image.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (H, W) where H and W are multiples of kernel_size.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits and patches.
        """
        patches = self._patches_from_image(data)
        param_binds = []
        for patch in patches:
            bind = {}
            for i, val in enumerate(patch):
                # Map pixel intensity [0, 1] to rotation angle
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute average probability of |1> per qubit
        total_ones = 0
        total_counts = 0
        for bitstring, cnt in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * cnt
            total_counts += cnt * self.n_qubits

        return total_ones / total_counts

    def forward(self, batch: np.ndarray) -> np.ndarray:
        """
        Apply the quantum filter to a batch of images.

        Parameters
        ----------
        batch : np.ndarray
            3‑D array of shape (B, H, W) where each image has dimensions
            divisible by kernel_size.

        Returns
        -------
        np.ndarray
            1‑D array of length B containing the activation for each image.
        """
        activations = []
        for img in batch:
            activations.append(self.run(img))
        return np.array(activations)

    def calibrate(self, images: np.ndarray, labels: np.ndarray):
        """
        Simple calibration routine that searches for a threshold that
        maximises the mean activation on a validation set.

        Parameters
        ----------
        images : np.ndarray
            Array of shape (B, H, W) of validation images.
        labels : np.ndarray
            Array of shape (B,) of ground‑truth labels (unused in this demo).
        """
        best_thr = self.threshold
        best_mean = -np.inf
        for thr in np.linspace(0, 1, 21):
            self.threshold = thr
            means = [self.run(img) for img in images]
            mean_val = np.mean(means)
            if mean_val > best_mean:
                best_mean = mean_val
                best_thr = thr
        self.threshold = best_thr
