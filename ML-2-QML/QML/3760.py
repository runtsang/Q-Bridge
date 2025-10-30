"""Quantum‑centric implementations used by the hybrid kernel."""
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

# --- Quantum kernel -------------------------------------------

class QuantumKernelAnsatz(tq.QuantumModule):
    """
    Encodes two classical feature vectors x and y using a reversible circuit.
    The overlap of the final state with |0...0> is returned as the kernel value.
    """
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Simple Ry‑rotation ansatz per wire
        self.ansatz = tq.QuantumModule()
        for i in range(self.n_wires):
            self.ansatz.add(tq.Ry(i))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        # Encode x
        for i, val in enumerate(x):
            q_device.apply(tq.Ry, wires=i, params=val)
        # Encode y in reverse (uncompute)
        for i, val in enumerate(y):
            q_device.apply(tq.Ry, wires=i, params=-val)

def quantum_kernel(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the kernel value via a quantum circuit.
    Parameters
    ----------
    x, y : np.ndarray
        Feature vectors of equal length.
    Returns
    -------
    float
        Overlap between the two encodings.
    """
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    q = QuantumKernelAnsatz()
    q.forward(q.q_device, x_t, y_t)
    amplitude = q.q_device.states.view(-1)[0]
    return float(torch.abs(amplitude))

# --- Quantum EstimatorQNN -------------------------------------

def EstimatorQNN():
    """
    Qiskit implementation of the EstimatorQNN example.
    Returns a qiskit_machine_learning.neural_networks.EstimatorQNN instance.
    """
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(theta, 0)
    qc.rx(theta, 0)
    observable = SparsePauliOp.from_list([("Y", 1)])
    estimator = StatevectorEstimator()
    return QiskitEstimatorQNN(circuit=qc,
                              observables=observable,
                              input_params=[theta],
                              weight_params=[theta],
                              estimator=estimator)

__all__ = ["QuantumKernelAnsatz", "quantum_kernel", "EstimatorQNN"]
