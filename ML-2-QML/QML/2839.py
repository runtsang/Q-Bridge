import numpy as np
import torch
from torch import nn
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers import Backend
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli
from qiskit import Aer

class QuantumEstimatorCircuit:
    """
    Wrapper around a single‑qubit parameterised circuit that returns the
    expectation value of the Y Pauli operator.  The circuit implements
    H‑gate, RY(input) and RX(weight) rotations, mirroring the original
    EstimatorQNN seed.  The circuit can be executed on a state‑vector
    simulator or any Aer backend, and exposes a run() method that
    accepts a NumPy array of input angles and returns a NumPy array of
    expectation values.
    """
    def __init__(self,
                 backend: Backend | None = None,
                 shots: int = 1024,
                 shift: float = np.pi / 2):
        self.n_qubits = 1
        self.shots = shots
        self.shift = shift

        # Build the parametrised circuit
        self.theta = Parameter("θ")
        self.weight = Parameter("w")

        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.rx(self.weight, 0)

        # Observable for Y expectation
        self.observable = Pauli(np.array([0, 1, 0], dtype=int))  # Y

        self.backend = backend or Aer.get_backend("aer_simulator_statevector")

        # Compile once
        self.compiled = transpile(self.circuit, self.backend)

    def run(self, inputs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """
        Execute the circuit for each input angle.  `inputs` is a 1‑D array
        of rotation angles.  Optional `weights` is a scalar or array of
        weight parameters; if omitted, the circuit uses the current
        weight parameter value.
        """
        if weights is None:
            weights = np.full_like(inputs, 0.0)
        else:
            weights = np.asarray(weights)

        # Prepare parameter bindings for each shot
        param_binds = []
        for inp, w in zip(inputs, weights):
            param_binds.append({self.theta: inp, self.weight: w})

        qobj = assemble(self.compiled,
                        shots=self.shots,
                        parameter_binds=param_binds)

        job = self.backend.run(qobj)
        result = job.result()

        # Compute expectation values
        exp_vals = []
        for counts in result.get_counts():
            total = sum(counts.values())
            exp = 0.0
            for bitstring, count in counts.items():
                val = int(bitstring, 2)
                eig = 1 if val == 0 else -1
                exp += eig * count / total
            exp_vals.append(exp)
        return np.array(exp_vals)

class HybridFunction(torch.autograd.Function):
    """
    Autograd wrapper that forwards a scalar input through the quantum
    circuit and returns the expectation value of Y.  Gradients are
    computed using the parameter‑shift rule for both the input and the
    weight parameter.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumEstimatorCircuit,
                shift: float, weight: torch.Tensor):
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.weight = weight

        inputs_np = inputs.detach().cpu().numpy()
        weight_np = weight.detach().cpu().numpy()
        exp = circuit.run(inputs_np, weights=np.full_like(inputs_np, weight_np.item()))
        out = torch.tensor(exp, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        weight = ctx.weight

        grad_inputs = []
        grad_weight = []

        for inp in inputs.detach().cpu().numpy():
            exp_plus = circuit.run(np.array([inp]), weights=np.array([weight.item() + shift]))
            exp_minus = circuit.run(np.array([inp]), weights=np.array([weight.item() - shift]))
            grad_inputs.append(exp_plus - exp_minus)

            exp_plus_w = circuit.run(np.array([inp]), weights=np.array([weight.item() + shift]))
            exp_minus_w = circuit.run(np.array([inp]), weights=np.array([weight.item() - shift]))
            grad_weight.append(exp_plus_w - exp_minus_w)

        grad_inputs = torch.tensor(grad_inputs, dtype=grad_output.dtype, device=grad_output.device)
        grad_weight = torch.tensor(grad_weight, dtype=grad_output.dtype, device=grad_output.device)

        return grad_inputs * grad_output, None, None, grad_weight * grad_output

class EstimatorQNN(nn.Module):
    """
    Hybrid estimator that takes the scalar output of the classical
    EstimatorQNN, feeds it into a parameterised quantum circuit, and
    returns the expectation value of Y as the final prediction.  The
    circuit is executed on the provided backend; by default it uses
    the Aer state‑vector simulator for deterministic results.  The
    class exposes a `weight` parameter that can be optimised jointly
    with the classical network.
    """
    def __init__(self,
                 backend=None,
                 shots: int = 1024,
                 shift: float = np.pi / 2):
        super().__init__()
        self.shift = shift
        self.quantum = QuantumEstimatorCircuit(
            backend=backend,
            shots=shots,
            shift=shift
        )
        # Register a trainable weight parameter
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x → quantum circuit → expectation value.
        The input `x` is expected to be a scalar tensor (output of the
        classical network).  The weight parameter is applied to the
        RX rotation of the circuit.
        """
        return HybridFunction.apply(x, self.quantum, self.shift, self.weight)

__all__ = ["QuantumEstimatorCircuit", "HybridFunction", "EstimatorQNN"]
