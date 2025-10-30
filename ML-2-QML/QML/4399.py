import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Sampler

def _self_attention_subcircuit(qr: QuantumRegister,
                               rotation_params: list[Parameter],
                               entangle_params: list[Parameter]) -> QuantumCircuit:
    """Build a tiny self‑attention style block using rotations and a single CNOT."""
    qc = QuantumCircuit(qr)
    for i in range(len(qr)):
        qc.rx(rotation_params[3*i], i)
        qc.ry(rotation_params[3*i+1], i)
        qc.rz(rotation_params[3*i+2], i)
    for i in range(len(qr)-1):
        qc.crx(entangle_params[i], i, i+1)
    return qc

class EstimatorQNNHybrid(QiskitEstimatorQNN):
    """Quantum EstimatorQNN that embeds a self‑attention subcircuit."""
    def __init__(self):
        n_qubits = 4

        # Input encoding parameters
        input_params = [Parameter(f"x{i}") for i in range(2)]

        # Variational parameters
        var_params = [Parameter(f"w{i}") for i in range(6)]

        # Self‑attention parameters
        rot_params = [Parameter(f"r{i}_{j}") for i in range(n_qubits) for j in range(3)]
        ent_params = [Parameter(f"e{i}") for i in range(n_qubits-1)]

        qr = QuantumRegister(n_qubits, "q")
        qc = QuantumCircuit(qr)

        # Encode inputs
        for i, p in enumerate(input_params):
            qc.ry(p, i)

        # Variational layer
        for i, p in enumerate(var_params):
            qc.rx(p, i)

        # Self‑attention subcircuit
        qc += _self_attention_subcircuit(qr, rot_params, ent_params)

        # Observable: tensor of Pauli‑Z on all qubits
        observable = SparsePauliOp.from_list([("Z", i) for i in range(n_qubits)])

        # Estimator primitive
        estimator = Sampler()

        super().__init__(
            circuit=qc,
            observables=observable,
            input_params=input_params,
            weight_params=var_params,
            estimator=estimator,
        )

def EstimatorQNN() -> EstimatorQNNHybrid:
    """Return a quantum estimator that mirrors the classical API."""
    return EstimatorQNNHybrid()

__all__ = ["EstimatorQNN", "EstimatorQNNHybrid"]
