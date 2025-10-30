import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class HybridConvAttention:
    """
    Quantum hybrid layer that first runs a quanvolution filter and then a
    quantum self‑attention block.  The output of the filter is encoded into
    rotation angles for the attention circuit, providing a fully quantum
    end‑to‑end pipeline.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the quanvolution filter (number of qubits = kernel_size**2).
    conv_shots : int, default 100
        Number of shots for the filter measurement.
    attention_shots : int, default 1024
        Number of shots for the attention circuit.
    conv_threshold : float, default 127
        Threshold for encoding classical data into rotation angles.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        conv_shots: int = 100,
        attention_shots: int = 1024,
        conv_threshold: float = 127,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.conv_shots = conv_shots
        self.attention_shots = attention_shots
        self.conv_threshold = conv_threshold

        # Backend
        self.backend = Aer.get_backend("qasm_simulator")

        # Build quanvolution circuit
        self._build_conv_circuit()

        # Parameters for the quantum self‑attention block
        self.attn_n_qubits = self.n_qubits
        self.attn_rot_params = np.random.uniform(0, 2 * np.pi, 3 * self.attn_n_qubits)
        self.attn_entangle_params = np.random.uniform(0, np.pi, self.attn_n_qubits - 1)

    def _build_conv_circuit(self):
        self._conv_circ = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._conv_circ.rx(self.theta[i], i)
        self._conv_circ.barrier()
        self._conv_circ += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self._conv_circ.measure_all()

    def _build_attn_circuit(self, rot_params: np.ndarray, ent_params: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.attn_n_qubits, "q")
        cr = ClassicalRegister(self.attn_n_qubits, "c")
        circ = QuantumCircuit(qr, cr)
        for i in range(self.attn_n_qubits):
            circ.rx(rot_params[3 * i], i)
            circ.ry(rot_params[3 * i + 1], i)
            circ.rz(rot_params[3 * i + 2], i)
        for i in range(self.attn_n_qubits - 1):
            circ.crx(ent_params[i], i, i + 1)
        circ.measure(qr, cr)
        return circ

    def run(self, data: np.ndarray) -> dict:
        """
        Execute the hybrid quantum pipeline on a single kernel‑sized patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        dict
            Measurement outcome frequencies from the attention circuit.
        """
        # Prepare parameter bindings for the quanvolution circuit
        param_binds = []
        for val in data.flatten():
            bind = {self.theta[i]: np.pi if val > self.conv_threshold else 0.0
                    for i in range(self.n_qubits)}
            param_binds.append(bind)

        # Run quanvolution
        job_conv = execute(self._conv_circ,
                           self.backend,
                           shots=self.conv_shots,
                           parameter_binds=param_binds)
        conv_counts = job_conv.result().get_counts(self._conv_circ)

        # Convert conv measurement statistics to rotation angles for attention
        angle_source = []
        for key, val in conv_counts.items():
            bits = [int(b) for b in key]
            angle_source.extend([np.pi if b else 0.0 for b in bits])
        if not angle_source:
            angle_source = [0.0] * self.attn_n_qubits
        rot_params = np.array(angle_source[:3 * self.attn_n_qubits])
        ent_params = self.attn_entangle_params

        # Build and run attention circuit
        attn_circ = self._build_attn_circuit(rot_params, ent_params)
        job_attn = execute(attn_circ, self.backend, shots=self.attention_shots)
        return job_attn.result().get_counts(attn_circ)

__all__ = ["HybridConvAttention"]
