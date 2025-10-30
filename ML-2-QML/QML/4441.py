import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler as StatevectorSampler, Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN

class QuantumSelfAttention(nn.Module):
    """
    Quantum implementation of a self‑attention block.
    Produces a probability vector over 2^n_qubits basis states.
    """
    def __init__(self, n_qubits: int, backend, shots: int = 1024):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots

        qr = QuantumRegister(n_qubits, "q")
        cr = ClassicalRegister(n_qubits, "c")
        self.circuit = QuantumCircuit(qr, cr)

        # Trainable rotation and entanglement parameters
        self.rotation_params = ParameterVector("rot", 3 * n_qubits)
        self.entangle_params = ParameterVector("ent", n_qubits - 1)

        for i in range(n_qubits):
            self.circuit.rx(self.rotation_params[3 * i], i)
            self.circuit.ry(self.rotation_params[3 * i + 1], i)
            self.circuit.rz(self.rotation_params[3 * i + 2], i)
        for i in range(n_qubits - 1):
            self.circuit.crx(self.entangle_params[i], i, i + 1)
        self.circuit.measure_all()

    def forward(self, rotation_vals: np.ndarray, entangle_vals: np.ndarray) -> torch.Tensor:
        """
        Execute the circuit with the given parameter values and return a
        probability vector over all basis states.
        """
        param_binds = {}
        for param, val in zip(self.rotation_params, rotation_vals):
            param_binds[param] = val
        for param, val in zip(self.entangle_params, entangle_vals):
            param_binds[param] = val

        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots, parameter_binds=[param_binds])
        job = self.backend.run(qobj)
        counts = job.result().get_counts()

        probs = np.zeros(2 ** self.n_qubits)
        for state, cnt in counts.items():
            idx = int(state[::-1], 2)
            probs[idx] = cnt / self.shots
        return torch.tensor(probs, dtype=torch.float32)

class UnifiedSamplerEstimatorAttentionQuantum(nn.Module):
    """
    Quantum‑centric counterpart of UnifiedSamplerEstimatorAttention.
    Replaces the classical sampler, estimator, and attention blocks with
    parameterized quantum circuits.  The final classification head
    remains classical for easy integration with PyTorch optimisers.
    """
    def __init__(self, backend, shots: int = 1024):
        super().__init__()
        self.backend = backend
        self.shots = shots

        # --- Sampler QNN ----------------------------------------------------
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc_sampler = QuantumCircuit(2)
        qc_sampler.ry(inputs[0], 0)
        qc_sampler.ry(inputs[1], 1)
        qc_sampler.cx(0, 1)
        qc_sampler.ry(weights[0], 0)
        qc_sampler.ry(weights[1], 1)
        qc_sampler.cx(0, 1)
        qc_sampler.ry(weights[2], 0)
        qc_sampler.ry(weights[3], 1)
        qc_sampler.measure_all()
        sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(circuit=qc_sampler,
                                      input_params=inputs,
                                      weight_params=weights,
                                      sampler=sampler)

        # --- Estimator QNN --------------------------------------------------
        params = [Parameter("input1"), Parameter("weight1")]
        qc_estimator = QuantumCircuit(1)
        qc_estimator.h(0)
        qc_estimator.ry(params[0], 0)
        qc_estimator.rx(params[1], 0)
        qc_estimator.measure_all()
        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(circuit=qc_estimator,
                                          observables=observable,
                                          input_params=[params[0]],
                                          weight_params=[params[1]],
                                          estimator=estimator)

        # --- Self‑attention circuit -----------------------------------------
        self.attention = QuantumSelfAttention(n_qubits=4,
                                              backend=self.backend,
                                              shots=self.shots)

        # --- Final classification head ------------------------------------
        # 2 (sampler) + 1 (estimator) + 16 (attention) = 19
        self.final = nn.Linear(2 + 1 + 16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Binary class probabilities of shape (batch, 2).
        """
        batch_size = x.size(0)

        # 1. Sampler: use x as input parameters, random weights for illustration
        input_params = x[:, :2]
        weight_params = torch.randn(batch_size, 4)
        sampler_out = []
        for inp, w in zip(input_params, weight_params):
            probs = self.sampler_qnn.forward([inp.tolist(), w.tolist()])
            sampler_out.append(torch.tensor(probs, dtype=torch.float32))
        sampler_out = torch.stack(sampler_out)  # (batch, 2)

        # 2. Estimator: use the first input dimension, random weight
        estimator_input = x[:, 0].unsqueeze(-1)
        estimator_weight = torch.randn(batch_size, 1)
        estimator_out = []
        for inp, w in zip(estimator_input, estimator_weight):
            val = self.estimator_qnn.forward([inp.tolist(), w.tolist()])
            estimator_out.append(torch.tensor(val, dtype=torch.float32))
        estimator_out = torch.stack(estimator_out).squeeze(-1)  # (batch,)

        # 3. Self‑attention: deterministic parameters for a baseline
        rotation_vals = np.zeros(4 * 3)
        entangle_vals = np.zeros(3)
        attention_out = self.attention.forward(rotation_vals, entangle_vals)  # (16,)
        attention_out = attention_out.unsqueeze(0).repeat(batch_size, 1)  # (batch, 16)

        # 4. Concatenate and classify
        combined = torch.cat([sampler_out,
                              estimator_out.unsqueeze(-1),
                              attention_out], dim=-1)
        logits = self.final(combined)
        probs = torch.softmax(logits, dim=-1)
        return probs

__all__ = ["UnifiedSamplerEstimatorAttentionQuantum"]
