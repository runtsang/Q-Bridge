from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
import torch
import torch.nn as nn
from typing import List, Tuple

def build_sampler_circuit(num_qubits: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector]]:
    """
    Build a quantum sampler circuit that maps two input parameters to a probability distribution.
    """
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    return qc, list(inputs), list(weights)

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Build a simple variational circuit similar to the quantum classifier in the seed.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class SamplerQNNHybrid(nn.Module):
    """
    Quantum implementation of the hybrid sampler‑classifier.
    """

    def __init__(
        self,
        input_dim: int = 2,
        sampler_hidden: int = 4,  # unused in quantum mode but kept for API compatibility
        classifier_output_dim: int = 2,
        num_qubits: int = 2,
        circuit_depth: int = 2,
        use_qlstm: bool = False,
        lstm_hidden_dim: int = 8,
        lstm_n_qubits: int = 4,
    ) -> None:
        super().__init__()
        # Quantum sampler
        sampler_circuit, sampler_inputs, sampler_weights = build_sampler_circuit(num_qubits)
        self.sampler = SamplerQNN(
            circuit=sampler_circuit,
            input_params=sampler_inputs,
            weight_params=sampler_weights,
            sampler=StatevectorSampler(),
        )
        # Quantum classifier
        self.classifier_circuit, self.input_params, self.weight_params, self.observables = build_classifier_circuit(num_qubits, circuit_depth)
        self.classifier = SamplerQNN(
            circuit=self.classifier_circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=StatevectorSampler(),
        )
        self.use_qlstm = use_qlstm
        if use_qlstm:
            # Placeholder: use a classical LSTM for demonstration
            self.lstm = nn.LSTM(input_size=2, hidden_size=lstm_hidden_dim, batch_first=True)
        else:
            self.lstm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Output probabilities of shape (batch, seq_len, classifier_output_dim).
        """
        batch, seq_len, _ = x.shape
        # Flatten to process each time step independently
        x_flat = x.reshape(batch * seq_len, -1)
        # Use the quantum sampler to get probability distributions
        sampler_probs = []
        for sample in x_flat:
            # sample is a torch tensor of shape (input_dim,)
            params = sample.detach().cpu().numpy().tolist()
            # Pad/truncate to match sampler input size (2)
            if len(params) < 2:
                params += [0.0] * (2 - len(params))
            elif len(params) > 2:
                params = params[:2]
            probs = self.sampler.run(params).reshape(-1)
            sampler_probs.append(probs)
        sampler_probs = torch.tensor(sampler_probs, dtype=torch.float32).reshape(batch, seq_len, 2)

        if self.lstm is not None:
            lstm_out, _ = self.lstm(sampler_probs)
            features = lstm_out
        else:
            features = sampler_probs

        # Use the quantum classifier
        classifier_probs = []
        for sample in features.reshape(batch * seq_len, -1):
            params = sample.detach().cpu().numpy().tolist()
            # Pad/truncate to match classifier input size
            if len(params) < len(self.input_params):
                params += [0.0] * (len(self.input_params) - len(params))
            elif len(params) > len(self.input_params):
                params = params[:len(self.input_params)]
            probs = self.classifier.run(params).reshape(-1)
            classifier_probs.append(probs)
        classifier_probs = torch.tensor(classifier_probs, dtype=torch.float32).reshape(batch, seq_len, -1)
        return classifier_probs
