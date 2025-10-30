# This QML module defines the quantum and photonic components used by the hybrid model.
# It is intentionally lightweight and relies on qiskit and strawberryfields for simulation.

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit.opflow import StateFn
from qiskit.quantum_info import Statevector

def build_quantum_encoder_circuit(latent_dim: int, reps: int = 3) -> QuantumCircuit:
    """
    Returns a parameterized RealAmplitudes circuit that will be bound to classical
    latent vectors during training.
    """
    qc = QuantumCircuit(latent_dim)
    ansatz = RealAmplitudes(latent_dim, reps=reps, insert_barriers=True)
    qc.compose(ansatz, inplace=True)
    return qc

def simulate_statevector(qc: QuantumCircuit, params: dict) -> Statevector:
    """
    Simulate the statevector of *qc* after binding the given parameters.
    """
    bound_qc = qc.bind_parameters(params)
    backend = AerSimulator(method='statevector')
    result = backend.run(bound_qc).result()
    return Statevector(result.get_statevector(bound_qc))

# Photonic fraud detection program
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

def build_photonic_fraud_program(layer_params: list[dict]) -> sf.Program:
    """
    Construct a Strawberry Fields program that mimics the fraud‑detection
    photonic circuit described in the reference.
    """
    prog = sf.Program(2)
    with prog.context as q:
        for params in layer_params:
            BSgate(params['bs_theta'], params['bs_phi']) | (q[0], q[1])
            for i, phase in enumerate(params['phases']):
                Rgate(phase) | q[i]
            for i, (r, phi) in enumerate(zip(params['squeeze_r'], params['squeeze_phi'])):
                Sgate(r, phi) | q[i]
            BSgate(params['bs_theta'], params['bs_phi']) | (q[0], q[1])
            for i, phase in enumerate(params['phases']):
                Rgate(phase) | q[i]
            for i, (r, phi) in enumerate(zip(params['displacement_r'], params['displacement_phi'])):
                Dgate(r, phi) | q[i]
            for i, k in enumerate(params['kerr']):
                Kgate(k) | q[i]
    return prog
