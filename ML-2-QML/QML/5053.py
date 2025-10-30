from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler

# Helper: quantum auto‑encoder circuit
def autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Variational ansatz – RealAmplitudes on the latent + trash qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    circuit.append(ansatz, range(num_latent + num_trash))

    circuit.barrier()
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])
    return circuit

# Helper: quantum self‑attention block
def self_attention_circuit(num_qubits: int,
                           rotation_params: np.ndarray | None = None,
                           entangle_params: np.ndarray | None = None) -> QuantumCircuit:
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "c")
    circuit = QuantumCircuit(qr, cr)

    if rotation_params is None:
        rotation_params = np.random.randn(3 * num_qubits)
    if entangle_params is None:
        entangle_params = np.random.randn(num_qubits - 1)

    for i in range(num_qubits):
        circuit.rx(rotation_params[3 * i], i)
        circuit.ry(rotation_params[3 * i + 1], i)
        circuit.rz(rotation_params[3 * i + 2], i)

    for i in range(num_qubits - 1):
        circuit.crx(entangle_params[i], i, i + 1)

    circuit.measure(qr, cr)
    return circuit

class HybridQuantumClassifier:
    """
    Quantum‑centric hybrid classifier that stitches a quantum auto‑encoder,
    a self‑attention block, and a SamplerQNN for probabilistic classification.
    """

    def __init__(self,
                 num_latent: int = 3,
                 num_trash: int = 2,
                 num_attention_qubits: int = 4,
                 backend=None) -> None:
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.num_attention_qubits = num_attention_qubits

        # Build sub‑circuits
        self.autoencoder = autoencoder_circuit(num_latent, num_trash)
        self.attention = self_attention_circuit(num_attention_qubits)

        # Compose the full circuit – the auto‑encoder feeds into the attention block
        self.circuit = QuantumCircuit(
            self.autoencoder.num_qubits + self.attention.num_qubits,
            name="hybrid_classifier"
        )
        self.circuit.compose(self.autoencoder, inplace=True)
        self.circuit.compose(self.attention, inplace=True)

        # SamplerQNN on top of the composed circuit
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],                      # no external inputs in this toy example
            weight_params=self.circuit.parameters, # variational parameters
            sampler=self.sampler,
            output_shape=2,                       # binary classification
        )

        # Classical optimizer for the variational parameters
        self.optimizer = COBYLA()

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Evaluate the quantum circuit on the backend and return a probability
        distribution over the two classes.

        Parameters
        ----------
        data : np.ndarray
            Feature matrix of shape (n_samples, n_features).  In this toy
            implementation the data is not explicitly mapped to parameters,
            but a real implementation would use a feature‑map.
        """
        # For each sample we run the circuit; here we return a single vector
        # because mapping from data to parameters is omitted.
        job = execute(self.circuit, self.backend, shots=1024)
        counts = job.result().get_counts(self.circuit)
        total = sum(counts.values())
        # Interpret the first qubit as the class label
        prob_0 = counts.get('0' * self.circuit.num_qubits, 0) / total
        prob_1 = 1.0 - prob_0
        return np.array([[prob_0, prob_1]])

    def train(self,
              data: np.ndarray,
              labels: np.ndarray,
              epochs: int = 5,
              learning_rate: float = 0.01) -> None:
        """
        Dummy training loop that optimizes the variational parameters of the
        auto‑encoder ansatz using COBYLA to minimise cross‑entropy loss.
        The mapping from classical data to circuit parameters is omitted for
        brevity – a real implementation would embed the data as rotation
        angles or via a feature‑map.
        """
        for _ in range(epochs):
            # Current prediction
            probs = self.predict(data)
            # Cross‑entropy loss (vectorised over samples)
            loss = -np.sum(labels * np.log(probs + 1e-8))
            # Objective for the optimizer
            def objective(flat_params):
                param_dict = {p: v for p, v in zip(self.circuit.parameters, flat_params)}
                self.circuit.assign_parameters(param_dict, inplace=True)
                probs = self.predict(data)
                return -np.sum(labels * np.log(probs + 1e-8))
            init_params = np.array([float(p) for p in self.circuit.parameters])
            self.optimizer.optimize(objective, init_params)

__all__ = ["HybridQuantumClassifier", "autoencoder_circuit", "self_attention_circuit"]
