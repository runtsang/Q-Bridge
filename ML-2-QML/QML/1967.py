from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as QSampler

class SamplerQNNGen162(SamplerQNN):
    """
    Quantum sampler with a deeper variational ansatz and
    dynamic sampling options. Extends the base SamplerQNN
    to support arbitrary depth and entanglement patterns.
    """
    def __init__(self,
                 num_qubits: int = 2,
                 layers: int = 4,
                 sampler: QSampler | None = None,
                 seed: int | None = None) -> None:
        self.num_qubits = num_qubits
        self.layers = layers
        self.seed = seed

        input_params = ParameterVector("input", num_qubits)
        weight_params = ParameterVector("weight", num_qubits * layers * 3)

        qc = QuantumCircuit(num_qubits)
        for l in range(layers):
            for q in range(num_qubits):
                qc.ry(input_params[q], q)
                qc.rz(weight_params[l * num_qubits + q], q)
            if l < layers - 1:
                for q in range(num_qubits - 1):
                    qc.cx(q, q + 1)
                qc.cx(num_qubits - 1, 0)

        if sampler is None:
            sampler = QSampler()
        if seed is not None:
            sampler.set_options(seed=seed)

        super().__init__(circuit=qc,
                         input_params=input_params,
                         weight_params=weight_params,
                         sampler=sampler)

    def forward(self, inputs: list[float]) -> list[float]:
        return super().forward(inputs)

    def sample(self, inputs: list[float], n_samples: int = 1) -> list[int]:
        return super().sample(inputs, n_samples)

__all__ = ["SamplerQNNGen162"]
