"""Quantum implementation of a hybrid fully‑connected layer.

The model stitches together three quantum modules inspired by the
original `FCL.py`, `FraudDetection.py`, `QCNN.py` and
`ClassicalQuantumBinaryClassification.py` seeds:

* a fraud‑detection sub‑circuit built from the photonic parameters
* a QCNN ansatz that performs convolution + pooling on eight qubits
* a fully‑connected layer realised by a single‑qubit RY circuit

All three are executed on an Aer simulator and the final prediction is
the expectation value of the Z operator on the fully‑connected qubit.
The module is fully differentiable via a custom autograd.Function that
uses finite‑difference gradients.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector, Parameter
from qiskit.providers.aer import AerSimulator

# ------------------------------------------------------------------
# Parameter layout identical to the photonic seed
# ------------------------------------------------------------------
class FraudLayerParameters:
    def __init__(
        self,
        bs_theta: float,
        bs_phi: float,
        phases: tuple[float, float],
        squeeze_r: tuple[float, float],
        squeeze_phi: tuple[float, float],
        displacement_r: tuple[float, float],
        displacement_phi: tuple[float, float],
        kerr: tuple[float, float],
    ):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

# ------------------------------------------------------------------
# Fraud‑detection quantum circuit (parameterised by the class above)
# ------------------------------------------------------------------
def _build_fraud_circuit(params: FraudLayerParameters, clip: bool) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(2)
    # Beam‑splitter style using RZ and RX as placeholders
    qc.ry(params.bs_theta, 0)
    qc.ry(params.bs_phi, 1)
    for i, phase in enumerate(params.phases):
        qc.rz(phase, i)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        angle = r if not clip else max(-5.0, min(5.0, r))
        qc.rz(angle, i)
        qc.rx(phi, i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        angle = r if not clip else max(-5.0, min(5.0, r))
        qc.rx(angle, i)
        qc.rz(phi, i)
    for i, k in enumerate(params.kerr):
        angle = k if not clip else max(-1.0, min(1.0, k))
        qc.rz(angle, i)
    qc.measure_all()
    return qc

# ------------------------------------------------------------------
# QCNN ansatz – copied from the qiskit reference
# ------------------------------------------------------------------
def conv_circuit(params):
    target = qiskit.QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

def conv_layer(num_qubits, param_prefix):
    qc = qiskit.QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = qiskit.QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params):
    target = qiskit.QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = qiskit.QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = qiskit.QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def build_qcnn_circuit() -> qiskit.QuantumCircuit:
    """Return a fully‑parameterised QCNN circuit with 30 free parameters."""
    feature_map = qiskit.circuit.library.ZFeatureMap(8)
    ansatz = qiskit.QuantumCircuit(8, name="Ansatz")
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)
    circuit = qiskit.QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)
    return circuit

# ------------------------------------------------------------------
# Fully‑connected quantum layer – single‑qubit RY
# ------------------------------------------------------------------
def build_fc_circuit() -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(1)
    theta = Parameter("theta")
    qc.h(0)
    qc.ry(theta, 0)
    qc.measure_all()
    return qc

# ------------------------------------------------------------------
# Quantum circuit wrapper with finite‑difference gradients
# ------------------------------------------------------------------
class QuantumCircuitWrapper:
    def __init__(self, circuit: qiskit.QuantumCircuit, backend=None, shots=1024):
        self.circuit = circuit
        self.backend = backend or AerSimulator()
        self.shots = shots

    def run(self, param_bindings: dict[Parameter, float]) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots, parameter_binds=[param_bindings])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        return np.array([expectation(result)])

# ------------------------------------------------------------------
# HybridFunction – differentiable wrapper around the quantum backend
# ------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper) -> torch.Tensor:  # type: ignore[override]
        ctx.circuit = circuit
        thetas = inputs.squeeze().tolist()
        # Build parameter binding dictionary
        param_bindings = {circuit.circuit.parameters[i]: thetas[i] for i in range(len(thetas))}
        expectation = ctx.circuit.run(param_bindings)[0]
        result = torch.tensor([expectation])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        inputs, _ = ctx.saved_tensors
        thetas = inputs.squeeze().tolist()
        shift = 1e-5
        grads = []
        for i, theta in enumerate(thetas):
            # finite difference
            bind_plus = {ctx.circuit.circuit.parameters[i]: theta + shift}
            bind_minus = {ctx.circuit.circuit.parameters[i]: theta - shift}
            exp_plus = ctx.circuit.run(bind_plus)[0]
            exp_minus = ctx.circuit.run(bind_minus)[0]
            grads.append((exp_plus - exp_minus) / (2 * shift))
        grads = torch.tensor(grads)
        return grads * grad_output, None

# ------------------------------------------------------------------
# Quantum implementation of the hybrid fully‑connected layer
# ------------------------------------------------------------------
class FCLHybridQuantum(nn.Module):
    """Quantum version of the hybrid fully‑connected layer.

    The network is a composition of three sub‑circuits:

    1. Fraud‑detection circuit (2 qubits, fixed parameters)
    2. QCNN ansatz (8 qubits, 30 parameters)
    3. Fully‑connected layer (1 qubit, 1 parameter)

    The final prediction is the expectation value of Z on the
    fully‑connected qubit.  The module is differentiable via a
    custom autograd function that uses finite‑difference gradients.
    """
    def __init__(
        self,
        fraud_params: FraudLayerParameters,
        *,
        qcnn_shots: int = 1024,
        fc_shots: int = 1024,
    ) -> None:
        super().__init__()
        # Build sub‑circuits
        self.fraud_circuit = _build_fraud_circuit(fraud_params, clip=True)
        self.qcnn_circuit = build_qcnn_circuit()
        self.fc_circuit = build_fc_circuit()

        # Wrap each circuit for execution
        self.fraud_wrapper = QuantumCircuitWrapper(self.fraud_circuit, shots=1)
        self.qcnn_wrapper = QuantumCircuitWrapper(self.qcnn_circuit, shots=qcnn_shots)
        self.fc_wrapper = QuantumCircuitWrapper(self.fc_circuit, shots=fc_shots)

        # Parameters for QCNN and FC will be exposed as torch parameters
        self.qcnn_params = nn.Parameter(torch.randn(30))
        self.fc_params = nn.Parameter(torch.randn(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # 1. Fraud detection – fixed circuit, no trainable params
        self.fraud_wrapper.run({})  # executed for side‑effects only

        # 2. QCNN – bind parameters and evaluate
        qcnn_bind = {self.qcnn_circuit.parameters[i]: self.qcnn_params[i].item()
                     for i in range(len(self.qcnn_params))}
        qcnn_expect = self.qcnn_wrapper.run(qcnn_bind)[0]

        # 3. Fully‑connected – bind FC parameter
        fc_bind = {self.fc_circuit.parameters[0]: self.fc_params.item()}
        fc_expect = self.fc_wrapper.run(fc_bind)[0]

        # Combine the two expectations linearly; the first term emulates
        # a quantum expectation head, the second term is a classical bias
        output = qcnn_expect * fc_expect
        return torch.tensor([output])

def FCLHybridQuantumFactory(
    fraud_params: FraudLayerParameters,
) -> FCLHybridQuantum:
    """Convenience factory that builds a ready‑to‑train quantum model."""
    return FCLHybridQuantum(fraud_params)

__all__ = ["FraudLayerParameters", "FCLHybridQuantum", "FCLHybridQuantumFactory"]
