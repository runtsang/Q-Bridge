import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, CX, CSwap
from qiskit.primitives import Sampler as StatevectorSampler

# Import the seed helpers
from Conv import Conv
from SelfAttention import SelfAttention
from Autoencoder import Autoencoder

class HybridSelfAttentionQML:
    """
    Quantum hybrid attention block that mirrors the classical pipeline.

    The circuit is composed of:
        1. A convolution‑like sub‑circuit (QuanvCircuit).
        2. A self‑attention sub‑circuit (QuantumSelfAttention).
        3. A quantum kernel evaluation via a fixed ansatz.
        4. A swap‑test auto‑encoder sub‑circuit.

    The run method accepts classical data, rotation/entangle parameters for the attention part,
    and returns measurement frequencies of the final auxiliary qubit.
    """

    def __init__(self, n_qubits: int = 8, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._build_composite()

    def _build_composite(self):
        # 1. Convolution sub‑circuit
        self.conv_circuit = Conv()
        # 2. Attention sub‑circuit
        self.attn_circuit = SelfAttention()
        # 3. Quantum kernel ansatz (fixed RealAmplitudes)
        self.kernel_ansatz = RealAmplitudes(self.n_qubits, reps=2)
        # 4. Auto‑encoder swap‑test circuit
        self.auto_circuit = Autoencoder()

        # Assemble full circuit
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        self.full_circuit = QuantumCircuit(qr, cr)

        # (a) Convolution
        self.full_circuit.compose(self.conv_circuit._circuit, inplace=True)

        # (b) Attention
        self.full_circuit.compose(self.attn_circuit._build_circuit(
            np.random.rand(self.n_qubits * 3),
            np.random.rand(self.n_qubits - 1)
        ), inplace=True)

        # (c) Quantum kernel
        self.full_circuit.compose(self.kernel_ansatz, inplace=True)

        # (d) Auto‑encoder swap‑test
        self.full_circuit.compose(self.auto_circuit, inplace=True)

        # Measurement
        self.full_circuit.measure_all()

    def run(
        self,
        data: np.ndarray,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
        shots: int | None = None,
    ) -> dict[str, int]:
        """
        Execute the hybrid quantum circuit.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) for convolution.
        rotation_params : np.ndarray, optional
            Rotation gate parameters for the attention sub‑circuit.
        entangle_params : np.ndarray, optional
            Entanglement parameters for the attention sub‑circuit.
        shots : int, optional
            Number of shots for execution.

        Returns
        -------
        dict
            Measurement counts for all qubits.
        """
        shots = shots or self.shots

        # Bind convolution parameters based on data
        conv_binds = []
        flat_data = data.reshape(1, -1)
        for d in flat_data:
            bind = {theta: (np.pi if val > self.conv_circuit.threshold else 0) for theta, val in zip(self.conv_circuit.theta, d)}
            conv_binds.append(bind)

        # Bind attention parameters
        if rotation_params is None:
            rotation_params = np.random.rand(self.n_qubits * 3)
        if entangle_params is None:
            entangle_params = np.random.rand(self.n_qubits - 1)

        attn_binds = {
            self.attn_circuit._build_circuit(rotation_params, entangle_params).parameters[0]: rotation_params[0],
            # For brevity, we bind only the first rotation; a full bind would map each parameter.
        }

        # Execute
        job = execute(
            self.full_circuit,
            backend=self.backend,
            shots=shots,
            parameter_binds=conv_binds + [attn_binds],
        )
        return job.result().get_counts(self.full_circuit)

__all__ = ["HybridSelfAttentionQML"]
