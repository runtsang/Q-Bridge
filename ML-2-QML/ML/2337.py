"""ConvGen module: hybrid classical‑quantum convolution filter.

The ConvGen class merges a 2‑D PyTorch convolution with a variational
quantum circuit.  It supports three modes:

* classic   – only the classical convolution is executed.
* quantum   – only the quantum circuit is executed.
* hybrid    – both are executed and the final output is an average of the two.

The class is a drop‑in replacement for the original Conv filter from the
anchor reference.  It can be used in any pipeline that expects a callable
object with a ``run`` method.
"""

import numpy as np
import torch
from torch import nn

# Optional import of qiskit; if unavailable, the quantum mode will raise
# an informative error.
try:
    import qiskit
    from qiskit.circuit.random import random_circuit
except ImportError as exc:  # pragma: no cover
    qiskit = None
    random_circuit = None

class ConvGen:
    """Hybrid classical‑quantum convolution filter."""

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        mode: str = "hybrid",
        shots: int = 100,
        backend: str | None = None,
    ):
        """
        Parameters
        ----------
        kernel_size : int
            Size of the 2‑D kernel (must match the input shape).
        threshold : float
            Threshold used in the sigmoid activation (classical) and in
            the quantum parameter binding.
        mode : {"classic", "quantum", "hybrid"}
            Execution mode.
        shots : int
            Number of shots for the quantum simulator.
        backend : str | None
            Name of the Qiskit backend.  If ``None`` the Aer qasm_simulator
            is used.
        """
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.mode = mode.lower()
        if self.mode not in {"classic", "quantum", "hybrid"}:
            raise ValueError(f"Unsupported mode {mode!r}")

        # Classical convolutional backbone
        self._classic = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Quantum circuit backbone (only if needed)
        if self.mode!= "classic":
            if qiskit is None:
                raise ImportError("qiskit is required for quantum or hybrid mode")
            self._n_qubits = kernel_size ** 2
            self._shots = shots
            self._threshold = threshold
            self._backend = (
                qiskit.Aer.get_backend(backend) if backend else qiskit.Aer.get_backend("qasm_simulator")
            )
            self._circuit = qiskit.QuantumCircuit(self._n_qubits)
            self._theta = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(self._n_qubits)]
            for i in range(self._n_qubits):
                self._circuit.rx(self._theta[i], i)
            self._circuit.barrier()
            # Add a random 2‑qubit layer
            self._circuit += random_circuit(self._n_qubits, 2)
            self._circuit.measure_all()

    def _run_classic(self, data: np.ndarray | torch.Tensor) -> float:
        """Run the classical convolution."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.astype(np.float32))
        # Ensure shape (1, 1, H, W)
        data = data.reshape(1, 1, self.kernel_size, self.kernel_size)
        logits = self._classic(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

    def _run_quantum(self, data: np.ndarray | torch.Tensor) -> float:
        """Run the quantum circuit."""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        # Flatten to 1‑D array of length n_qubits
        flat = data.reshape(1, self._n_qubits)
        param_binds = []
        for sample in flat:
            bind = {}
            for i, val in enumerate(sample):
                # Map the raw value to a rotation angle via a simple linear map
                bind[self._theta[i]] = np.pi if val > self._threshold else 0.0
            param_binds.append(bind)
        job = qiskit.execute(
            self._circuit,
            self._backend,
            shots=self._shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self._circuit)
        # Compute average probability of measuring |1> over all qubits
        total_ones = 0
        total_counts = 0
        for bitstring, cnt in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * cnt
            total_counts += cnt
        return total_ones / (total_counts * self._n_qubits)

    def run(self, data: np.ndarray | torch.Tensor) -> float:
        """Execute the filter in the selected mode."""
        if self.mode == "classic":
            return self._run_classic(data)
        if self.mode == "quantum":
            return self._run_quantum(data)
        # hybrid: average of classical and quantum outputs
        classic_out = self._run_classic(data)
        quantum_out = self._run_quantum(data)
        return 0.5 * classic_out + 0.5 * quantum_out

    # --------------------------------------------------------------------- #
    #  Additional utilities inspired by GraphQNN
    # --------------------------------------------------------------------- #
    def fidelity_graph(
        self,
        data_list: list[np.ndarray | torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ):
        """
        Build a weighted graph from the outputs of the filter for a list of
        inputs.  The edge weights are based on the absolute difference between
        outputs, mirroring the fidelity‑based adjacency in the GraphQNN
        reference.

        Parameters
        ----------
        data_list : list
            List of input arrays.
        threshold : float
            Primary similarity threshold.
        secondary : float | None
            Secondary threshold for weaker connections.
        secondary_weight : float
            Weight assigned to secondary edges.

        Returns
        -------
        networkx.Graph
            Graph where nodes correspond to the inputs and edges are added
            when the output similarity satisfies the thresholds.
        """
        import networkx as nx

        outputs = [self.run(d) for d in data_list]
        G = nx.Graph()
        G.add_nodes_from(range(len(outputs)))
        for i, out_i in enumerate(outputs):
            for j, out_j in enumerate(outputs):
                if j <= i:
                    continue
                # similarity as 1 - abs difference (since outputs in [0,1])
                sim = 1.0 - abs(out_i - out_j)
                if sim >= threshold:
                    G.add_edge(i, j, weight=1.0)
                elif secondary is not None and sim >= secondary:
                    G.add_edge(i, j, weight=secondary_weight)
        return G

__all__ = ["ConvGen"]
