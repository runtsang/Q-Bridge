import torch
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable, List, Tuple

class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Random two‑qubit quantum kernel applied to 2×2 image patches."""
    def __init__(self, n_wires: int = 4, n_ops: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [x[:, r, c], x[:, r, c+1], x[:, r+1, c], x[:, r+1, c+1]],
                    dim=1
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[Parameter], List[Parameter], List[SparsePauliOp]]:
    """Construct a variational ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
    observables = [SparsePauliOp("I"*i + "Z" + "I"*(num_qubits-i-1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters]
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

def _apply_layer(modes: Iterable, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class EstimatorQNNQuantum:
    """Quantum estimator neural network using Qiskit."""
    def __init__(self):
        params1 = [Parameter("input1"), Parameter("weight1")]
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc1.ry(params1[0], 0)
        qc1.rx(params1[1], 0)
        observable1 = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        self.qnn = QiskitEstimatorQNN(
            circuit=qc1,
            observables=observable1,
            input_params=[params1[0]],
            weight_params=[params1[1]],
            estimator=estimator
        )

    def evaluate(self, inputs: List[float]) -> List[float]:
        return self.qnn.evaluate(inputs)

__all__ = [
    "QuanvolutionFilterQuantum",
    "build_classifier_circuit",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "EstimatorQNNQuantum",
]
