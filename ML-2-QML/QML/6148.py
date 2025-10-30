import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import AerSimulator

class HybridQLSTMAttention:
    """
    Quantum‑centric hybrid model implemented with Qiskit.
    The forward method runs a quantum self‑attention circuit followed
    by a quantum LSTM where each gate is a small 2‑qubit circuit.
    """
    def __init__(self, seq_len: int, n_qubits_attn: int = 8, n_qubits_lstm: int = 4, shots: int = 1024):
        self.seq_len = seq_len
        self.n_qubits_attn = max(seq_len, n_qubits_attn)
        self.n_qubits_lstm = n_qubits_lstm
        self.shots = shots
        self.sim = AerSimulator()

    def _attention_circuit(self, rotations, entangles):
        """
        Build a circuit that encodes a sequence of tokens into qubits,
        applies rotations and entanglement, and measures Pauli‑Z.
        """
        n = len(rotations)
        qr = QuantumRegister(n, 'q')
        cr = ClassicalRegister(n, 'c')
        circ = QuantumCircuit(qr, cr)
        # Map each token to a rotation on its qubit
        for i, ang in enumerate(rotations):
            circ.rx(ang, qr[i])
        # Entangle adjacent qubits with CRX gates
        for i, ang in enumerate(entangles):
            circ.crx(ang, qr[i], qr[(i + 1) % n])
        circ.measure(qr, cr)
        return circ

    def _run_circuit(self, circ):
        """Execute the circuit and return a probability distribution over qubit strings."""
        job = self.sim.run(circ, shots=self.shots)
        result = job.result()
        counts = result.get_counts(circ)
        probs = np.array([counts.get(format(i, f'0{circ.num_qubits}b'), 0) for i in range(2 ** circ.num_qubits)])
        probs = probs / probs.sum()
        return probs

    def _gate_circuit(self, name: str):
        """
        Small 2‑qubit circuit for a single LSTM gate.
        """
        qr = QuantumRegister(2, f'{name}_q')
        cr = ClassicalRegister(2, f'{name}_c')
        circ = QuantumCircuit(qr, cr)
        # Simple rotations
        circ.rx(0.3, qr[0])
        circ.ry(0.4, qr[1])
        circ.cx(qr[0], qr[1])
        circ.measure(qr, cr)
        return circ

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """
        input_ids: numpy array of shape (batch, seq_len) integer token ids
        Returns logits: numpy array of shape (batch, 1)
        """
        batch, seq_len = input_ids.shape
        logits = []
        for b in range(batch):
            # Map token ids to rotation angles (simple linear mapping)
            rotations = (input_ids[b] / 10.0) * np.pi
            entangles = np.full(seq_len - 1, np.pi / 4)
            attn_circ = self._attention_circuit(rotations, entangles)
            probs = self._run_circuit(attn_circ)
            # Convert probabilities to attention weights over tokens
            attn_weights = probs[:seq_len]
            attn_weights = attn_weights / attn_weights.sum()
            # Weighted token sum as representation
            rep = np.sum(input_ids[b] * attn_weights)
            # Build LSTM gates
            f_circ = self._gate_circuit('forget')
            i_circ = self._gate_circuit('input')
            g_circ = self._gate_circuit('update')
            o_circ = self._gate_circuit('output')
            f = self._run_circuit(f_circ)[0]
            i = self._run_circuit(i_circ)[0]
            g = self._run_circuit(g_circ)[0]
            o = self._run_circuit(o_circ)[0]
            # Simple hidden state update using scalar gate values
            hx = o * np.tanh(g)
            logits.append([hx])
        return np.array(logits)
