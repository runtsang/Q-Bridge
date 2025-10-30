import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.opflow import StateFn, AerPauliSumOp
