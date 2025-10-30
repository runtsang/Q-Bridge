import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Iterable, List

class SelfAttention(nn.Module):
    """
    Quantum self‑attention with optional quantum LSTM gating and quantum classifier.
    """
    def __init__(
        self,
        n_qubits: int,
        embed_dim: int,
        use_lstm: bool = False,
        lstm_wires: int = 4,
        classifier_depth: int = 2,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.embed_dim = embed_dim
        self.use_lstm = use_lstm

        # Attention circuit parameters
        self.rotation_params = ParameterVector("rot", 3 * n_qubits)
        self.entangle_params = ParameterVector("ent", n_qubits - 1)
        self.attn_circ = self._build_attention_circuit()

        # Optional quantum LSTM gating
        if use_lstm:
            self.lstm = self._build_quantum_lstm(lstm_wires)

        # Classifier circuit
        self.classifier_circ, self.enc_params, self.weight_params, self.observables = \
            self._build_classifier_circuit(n_qubits, classifier_depth)

        # Simulation backend
        self.backend = AerSimulator()

    def _build_attention_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circ = QuantumCircuit(qr, cr)

        # Encode input via RX,RY,RZ rotations
        for i in range(self.n_qubits):
            circ.rx(self.rotation_params[3 * i], qr[i])
            circ.ry(self.rotation_params[3 * i + 1], qr[i])
            circ.rz(self.rotation_params[3 * i + 2], qr[i])

        # Entangling layer
        for i in range(self.n_qubits - 1):
            circ.cx(qr[i], qr[i + 1])

        circ.measure(qr, cr)
        return circ

    def _build_quantum_lstm(self, wires: int) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList(
                    [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
                self.encoder(qdev, x)
                for wire, gate in enumerate(self.params):
                    gate(qdev, wires=wire)
                for wire in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[wire, wire + 1])
                return self.measure(qdev)

        return QLayer(wires)

    def _build_classifier_circuit(
        self,
        num_qubits: int,
        depth: int,
    ) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circ = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            circ.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circ.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                circ.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return circ, [encoding], [weights], observables

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_values: np.ndarray,
        entangle_values: np.ndarray,
        classifier_values: np.ndarray,
        shots: int = 1024,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Placeholder to keep the signature compatible with the classical version.
        rotation_values : np.ndarray
            Rotation angles for the attention circuit (shape (3 * n_qubits,)).
        entangle_values : np.ndarray
            Entangling gate angles for the attention circuit
            (shape (n_qubits - 1,)).
        classifier_values : np.ndarray
            Variational parameters for the classifier circuit
            (shape (num_qubits * depth,)).
        shots : int
            Number of measurement shots.

        Returns
        -------
        logits : torch.Tensor
            Tensor of shape (2,) containing the two logits for binary classification.
        """
        # Bind attention parameters
        param_map = {str(p): val for p, val in zip(self.rotation_params, rotation_values)}
        param_map.update({str(p): val for p, val in zip(self.entangle_params, entangle_values)})
        bound_attn = self.attn_circ.bind_parameters(param_map)

        # Execute attention circuit
        qobj_attn = transpile(bound_attn, self.backend)
        job_attn = self.backend.run(qobj_attn, shots=shots)
        counts_attn = job_attn.result().get_counts(bound_attn)

        # Convert counts to expectation values of Z
        exp_z = np.zeros(self.n_qubits)
        for bitstring, cnt in counts_attn.items():
            bits = [int(b) for b in reversed(bitstring)]
            z_vals = 1 - 2 * np.array(bits)  # 0->+1, 1->-1
            exp_z += cnt * z_vals
        exp_z /= shots

        # Optional quantum LSTM gating
        if self.use_lstm:
            attn_tensor = torch.tensor(exp_z, dtype=torch.float32).unsqueeze(0)
            gated = self.lstm(attn_tensor)  # shape (1, n_qubits)
            exp_z = gated.squeeze(0).detach().numpy()

        # Build classifier circuit with attention expectation as encoding
        param_map_cls = {str(p): val for p, val in zip(self.enc_params[0], exp_z)}
        param_map_cls.update({str(p): val for p, val in zip(self.weight_params[0], classifier_values)})
        bound_cls = self.classifier_circ.bind_parameters(param_map_cls)

        # Execute classifier circuit
        qobj_cls = transpile(bound_cls, self.backend)
        job_cls = self.backend.run(qobj_cls, shots=shots)
        counts_cls = job_cls.result().get_counts(bound_cls)

        # Convert measurement counts to logits
        logits = np.zeros(2)
        for bitstring, cnt in counts_cls.items():
            bits = [int(b) for b in reversed(bitstring)]
            logits[0] += cnt * (1 if bits[0] == 0 else -1)
            logits[1] += cnt * (1 if bits[1] == 0 else -1)
        logits /= shots
        return torch.tensor(logits, dtype=torch.float32)

__all__ = ["SelfAttention"]
