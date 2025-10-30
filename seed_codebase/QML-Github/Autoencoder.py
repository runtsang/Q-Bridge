import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import SamplerQNN


def Autoencoder():
    algorithm_globals.random_seed = 42
    sampler = Sampler()
    def ansatz(num_qubits):
        return RealAmplitudes(num_qubits, reps=5)

    def auto_encoder_circuit(num_latent, num_trash):
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()
        auxiliary_qubit = num_latent + 2 * num_trash
        # swap test
        circuit.h(auxiliary_qubit)
        for i in range(num_trash):
            circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)

        circuit.h(auxiliary_qubit)
        circuit.measure(auxiliary_qubit, cr[0])
        return circuit


    num_latent = 3
    num_trash = 2
    circuit = auto_encoder_circuit(num_latent, num_trash)
    circuit.draw(output="mpl", style="clifford")

    def domain_wall(circuit, a, b):
        # Here we place the Domain Wall to qubits a - b in our circuit
        for i in np.arange(int(b / 2), int(b)):
            circuit.x(i)
        return circuit


    domain_wall_circuit = domain_wall(QuantumCircuit(5), 0, 5)
    domain_wall_circuit.draw("mpl", style="clifford")

    ae = auto_encoder_circuit(num_latent, num_trash)
    qc = QuantumCircuit(num_latent + 2 * num_trash + 1, 1)
    qc = qc.compose(domain_wall_circuit, range(num_latent + num_trash))
    qc = qc.compose(ae)
    qc.draw(output="mpl", style="clifford")

    # Here we define our interpret for our SamplerQNN
    def identity_interpret(x):
        return x

    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=ae.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn