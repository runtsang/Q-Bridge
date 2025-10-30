import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliSumOp

class EstimatorQNNModel:
    """Variational quantum neural network that encodes a vector of
    real features into rotation angles on a single qubit, applies
    a stack of parameterised RX‑rotations, and measures the
    expectation value of the Y Pauli operator.  The circuit is
    built for the Aer state‑vector simulator and exposes a
    ``forward`` method that returns the expectation value.  A
    simple parameter‑shift gradient is also provided."""
    def __init__(self,
                 input_dim: int = 2,
                 n_layers: int = 3,
                 backend_name: str = "aer_simulator_statevector"):
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.backend = Aer.get_backend(backend_name)

        # Input encoding parameters
        self.input_params = ParameterVector("x", input_dim)

        # Variational parameters
        self.weight_params = ParameterVector("w", n_layers * input_dim)

        # Build circuit
        self.circuit = QuantumCircuit(1)
        # Encode inputs
        for param in self.input_params:
            self.circuit.ry(param, 0)
        # Variational layers
        for _ in range(n_layers):
            for w in self.weight_params:
                self.circuit.rx(w, 0)
            # Self‑entangling placeholder (CZ with same qubit)
            self.circuit.cz(0, 0)
        # Observable
        self.observable = PauliSumOp.from_list([("Y", 1)])

    def forward(self, inputs: np.ndarray) -> float:
        """Return the expectation value of Y for the given input."""
        bound = {p: v for p, v in zip(self.input_params, inputs)}
        bound.update({p: 0.0 for p in self.weight_params})
        bound_circ = self.circuit.bind_parameters(bound)
        result = execute(bound_circ, self.backend, shots=1024).result()
        state = result.get_statevector(bound_circ)
        y_matrix = self.observable.to_matrix()
        exp_val = np.real(np.vdot(state, y_matrix @ state))
        return float(exp_val)

    def gradient(self, inputs: np.ndarray) -> np.ndarray:
        """Parameter‑shift gradient w.r.t. the variational weights."""
        grads = np.zeros(len(self.weight_params))
        shift = np.pi / 2
        for idx, param in enumerate(self.weight_params):
            # Positive shift
            bound_plus = {p: v for p, v in zip(self.input_params, inputs)}
            bound_plus.update({p: shift if p == param else 0.0 for p in self.weight_params})
            circ_plus = self.circuit.bind_parameters(bound_plus)
            exp_plus = self._expectation_from_circuit(circ_plus)

            # Negative shift
            bound_minus = {p: v for p, v in zip(self.input_params, inputs)}
            bound_minus.update({p: -shift if p == param else 0.0 for p in self.weight_params})
            circ_minus = self.circuit.bind_parameters(bound_minus)
            exp_minus = self._expectation_from_circuit(circ_minus)

            grads[idx] = 0.5 * (exp_plus - exp_minus)
        return grads

    def _expectation_from_circuit(self, circ: QuantumCircuit) -> float:
        result = execute(circ, self.backend, shots=1024).result()
        state = result.get_statevector(circ)
        y_matrix = self.observable.to_matrix()
        return float(np.real(np.vdot(state, y_matrix @ state)))

def EstimatorQNN(**kwargs) -> EstimatorQNNModel:
    """Convenience constructor mirroring the original API."""
    return EstimatorQNNModel(**kwargs)

__all__ = ["EstimatorQNN"]
