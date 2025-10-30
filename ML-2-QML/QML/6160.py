"""Hybrid quantum estimator that combines a Qiskit parameterised circuit with a photonic‑style sub‑circuit."""
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp

def HybridEstimatorQNN() -> EstimatorQNN:
    # Base circuit parameters
    input_param = Parameter("input1")
    weight_param = Parameter("weight1")

    base_circuit = QuantumCircuit(1)
    base_circuit.h(0)
    base_circuit.ry(input_param, 0)
    base_circuit.rx(weight_param, 0)

    # Photonic sub‑circuit parameters
    bs_theta = Parameter("bs_theta")
    bs_phi = Parameter("bs_phi")
    phase0 = Parameter("phase0")
    phase1 = Parameter("phase1")
    squeeze_r0 = Parameter("squeeze_r0")
    squeeze_r1 = Parameter("squeeze_r1")
    squeeze_phi0 = Parameter("squeeze_phi0")
    squeeze_phi1 = Parameter("squeeze_phi1")
    disp_r0 = Parameter("disp_r0")
    disp_r1 = Parameter("disp_r1")
    disp_phi0 = Parameter("disp_phi0")
    disp_phi1 = Parameter("disp_phi1")
    kerr0 = Parameter("kerr0")
    kerr1 = Parameter("kerr1")

    sub_circuit = QuantumCircuit(2)
    # Beam splitter approximation
    sub_circuit.cx(0, 1)
    sub_circuit.cx(1, 0)
    # Phase rotations
    sub_circuit.rz(phase0, 0)
    sub_circuit.rz(phase1, 1)
    # Squeezing approximation
    sub_circuit.rx(squeeze_r0, 0)
    sub_circuit.rx(squeeze_r1, 1)
    # Displacement approximation
    sub_circuit.rz(disp_phi0, 0)
    sub_circuit.rz(disp_phi1, 1)
    # Kerr approximation
    sub_circuit.rz(kerr0, 0)
    sub_circuit.rz(kerr1, 1)

    # Merge circuits into a 3‑qubit circuit
    full_circuit = QuantumCircuit(3)
    full_circuit.compose(base_circuit, qubits=[0], inplace=True)
    full_circuit.compose(sub_circuit, qubits=[1, 2], inplace=True)

    # Observable
    observable = SparsePauliOp.from_list([("Y" * full_circuit.num_qubits, 1)])

    # Estimator
    estimator = Estimator()
    estimator_qnn = EstimatorQNN(
        circuit=full_circuit,
        observables=observable,
        input_params=[input_param],
        weight_params=[
            weight_param,
            bs_theta, bs_phi,
            phase0, phase1,
            squeeze_r0, squeeze_r1, squeeze_phi0, squeeze_phi1,
            disp_r0, disp_r1, disp_phi0, disp_phi1,
            kerr0, kerr1
        ],
        estimator=estimator,
    )
    return estimator_qnn

__all__ = ["HybridEstimatorQNN"]
