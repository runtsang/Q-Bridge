from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
import numpy as np

class SamplerQNN__gen354:
    """
    Quantum sampler network with 3 qubits and two variational layers.
    Extends the baseline QNN by adding entanglement across all qubits
    and a second layer of RZ rotations.
    """

    def __init__(self, input_dim: int = 2, weight_dim: int = 6, qubit_count: int = 3):
        self.input_params = ParameterVector("input", input_dim)
        self.weight_params = ParameterVector("weight", weight_dim)
        self.qubit_count = qubit_count
        self.circuit = self._build_circuit()
        self.sampler = Sampler()
        self.qnn = QiskitSamplerQNN(circuit=self.circuit,
                                    input_params=self.input_params,
                                    weight_params=self.weight_params,
                                    sampler=self.sampler)

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.qubit_count)
        # Input rotations on the first two qubits
        qc.ry(self.input_params[0], 0)
        qc.ry(self.input_params[1], 1)
        # First variational layer: Y rotations
        for i in range(self.qubit_count):
            qc.ry(self.weight_params[i], i)
        # Entangling pattern across all qubits
        qc.cx(0, 1)
        qc.cx(1, 2)
        # Second variational layer: Z rotations
        for i in range(self.qubit_count):
            qc.rz(self.weight_params[self.qubit_count + i], i)
        return qc

    def sample(self, inputs: np.ndarray, shots: int = 1024, weight_values: np.ndarray | None = None):
        """
        Execute the QNN for the provided classical inputs and optional
        weight values.  If weight_values is None the current parameters
        of the SamplerQNN are used.
        """
        # Map input parameters
        bound_inputs = {param: val for param, val in zip(self.input_params, inputs)}
        # Map weight parameters
        if weight_values is None:
            weight_values = [0.0] * len(self.weight_params)
        bound_weights = {param: val for param, val in zip(self.weight_params, weight_values)}
        # Bind all parameters to the circuit
        bound_circuit = self.circuit.bind_parameters({**bound_inputs, **bound_weights})
        # Execute using the Qiskit Sampler primitive
        return self.sampler.run(bound_circuit, shots=shots).result().get_counts()
