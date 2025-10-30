"""Hybrid quantum estimator with classical auto‑encoder style encoding.

The circuit combines:
* a RealAmplitudes feature map that acts as an auto‑encoder‑like block,
* a random layer for expressivity, and
* a swap‑test style read‑out that yields a scalar expectation value.

The function returns a qiskit_machine_learning.neural_networks.EstimatorQNN
which can be trained with any of Qiskit's primitives.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, Random
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StateEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter
import numpy as np

def EstimatorQNNGen083(
    num_features: int,
    num_latent: int = 3,
    num_trash: int = 2,
) -> EstimatorQNN:
    """
    Build a quantum neural network that mimics the hybrid classical‑quantum
    estimator defined in the ML counterpart.

    Parameters
    ----------
    num_features : int
        Dimensionality of the raw input vector.
    num_latent : int
        Size of the latent space used by the classical auto‑encoder part.
    num_trash : int
        Number of ancillary qubits used for the swap‑test style read‑out.
    """
    # Feature map that acts like an auto‑encoder: RealAmplitudes on latent qubits
    def feature_map(latent_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(latent_qubits, reps=5)

    # Random layer to increase expressivity
    def random_layer(wires: int) -> QuantumCircuit:
        return Random(num_qubits=wires, reps=3)

    # Construct the auto‑encoder‑style circuit
    def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode latent part
        qc.compose(feature_map(num_latent), range(0, num_latent), inplace=True)
        qc.barrier()

        # Swap‑test style read‑out on ancillary qubits
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    # Full circuit: feature map + random layer + auto‑encoder read‑out
    circuit = auto_encoder_circuit(num_latent, num_trash)
    # Add random layer on all qubits except the measurement qubit
    random_qc = random_layer(num_latent + 2 * num_trash)
    circuit.compose(random_qc, range(0, num_latent + 2 * num_trash), inplace=True)

    # Define observable: expectation of PauliZ on the measurement qubit
    observable = SparsePauliOp.from_list([("Z" * circuit.num_qubits, 1)])

    # Parameters: no input parameters (classical auto‑encoder handles raw data)
    input_params = []
    weight_params = random_qc.parameters

    # Estimator that evaluates expectation values
    estimator = StateEstimator()

    return EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )

__all__ = ["EstimatorQNNGen083"]
