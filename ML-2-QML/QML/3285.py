"""Quantum counterpart of the hybrid classifier autoencoder.

Provides a RealAmplitudes variational autoencoder that compresses classical data
into a latent sub‑space, and a subsequent classification ansatz that consumes
the latent vector.  The circuits are compatible with the classical
`HybridClassifierAutoencoder` API via a simple parameter‑vector interface,
enabling end‑to‑end hybrid training with Qiskit Machine Learning primitives.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.primitives import StatevectorSampler
from qiskit_machine_learning.utils import algorithm_globals

# Ensure reproducibility
algorithm_globals.random_seed = 42


def build_quantum_autoencoder(num_qubits: int,
                              reps: int = 3,
                              trash: int = 0) -> Tuple[QuantumCircuit, Iterable, Iterable]:
    """
    Construct a quantum autoencoder that maps *num_qubits* input qubits to a
    compressed latent space of size *num_qubits - trash*.

    Parameters
    ----------
    num_qubits: int
        Total number of qubits including latent and trash.
    reps: int
        Number of RealAmplitudes repetitions.
    trash: int
        Number of trash qubits to be discarded after encoding.

    Returns
    -------
    circuit: QuantumCircuit
        The variational autoencoder circuit.
    encoding_params: ParameterVector
        Classical parameters that encode the raw input features.
    weight_params: ParameterVector
        Variational parameters of the RealAmplitudes ansatz.
    """
    latent = num_qubits - trash
    encoding = ParameterVector('x', num_qubits)
    weights = ParameterVector('theta', latent * reps)
    qc = QuantumCircuit(num_qubits)

    # Data encoding
    for qubit, param in zip(range(num_qubits), encoding):
        qc.rx(param, qubit)

    # Variational block
    qc.append(RealAmplitudes(num_qubits, reps=reps), range(num_qubits))

    # Swap test to discard trash qubits
    if trash:
        aux = num_qubits
        qc.add_register(QuantumRegister(1, 'aux'))
        qc.h(aux)
        for i in range(trash):
            qc.cswap(aux, i, latent + i)
        qc.h(aux)
        qc.measure(aux, ClassicalRegister(1, 'c'))

    return qc, encoding, weights


def build_quantum_classifier(num_latent: int,
                             depth: int = 2,
                             observables: Iterable[SparsePauliOp] | None = None) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Build a variational classifier that consumes a latent vector encoded on
    *num_latent* qubits.  The circuit uses a simple alternating Ry–CNOT
    ansatz and measures Z on each qubit.

    Parameters
    ----------
    num_latent: int
        Number of qubits that carry the latent representation.
    depth: int
        Number of layers in the variational ansatz.
    observables: Iterable[SparsePauliOp], optional
        Custom observables; if None, default to Z on each qubit.

    Returns
    -------
    circuit: QuantumCircuit
        The classification circuit.
    encoding: ParameterVector
        Parameters that encode the latent vector.
    weights: ParameterVector
        Variational parameters.
    observables: list[SparsePauliOp]
        Observables measured to produce the output.
    """
    encoding = ParameterVector('z', num_latent)
    weights = ParameterVector('phi', num_latent * depth)
    qc = QuantumCircuit(num_latent)

    # Encode latent vector
    for qubit, param in zip(range(num_latent), encoding):
        qc.rx(param, qubit)

    # Variational ansatz
    idx = 0
    for _ in range(depth):
        for q in range(num_latent):
            qc.ry(weights[idx], q)
            idx += 1
        for q in range(num_latent - 1):
            qc.cz(q, q + 1)

    # Observables
    if observables is None:
        observables = [SparsePauliOp('I' * i + 'Z' + 'I' * (num_latent - i - 1))
                       for i in range(num_latent)]

    return qc, encoding, weights, observables


def build_sampler_qnn(circuit: QuantumCircuit,
                      encoding: Iterable,
                      weights: Iterable,
                      observables: Iterable[SparsePauliOp],
                      interpret=None,
                      output_shape: int | None = None) -> SamplerQNN:
    """
    Wrap a circuit in a Qiskit Machine Learning SamplerQNN for easy training.

    Parameters
    ----------
    circuit: QuantumCircuit
    encoding: Iterable[Parameter]
        Classical data encoding parameters.
    weights: Iterable[Parameter]
        Variational parameters.
    observables: Iterable[SparsePauliOp]
    interpret: Callable[[np.ndarray], np.ndarray], optional
        Function that maps raw measurement probabilities to model output.
    output_shape: int, optional
        Desired output shape if interpret is None.

    Returns
    -------
    SamplerQNN
    """
    if interpret is None:
        interpret = lambda x: x.reshape(output_shape) if output_shape else x
    sampler = StatevectorSampler()
    return SamplerQNN(circuit=circuit,
                      input_params=list(encoding),
                      weight_params=list(weights),
                      interpret=interpret,
                      output_shape=output_shape,
                      sampler=sampler)


__all__ = [
    "build_quantum_autoencoder",
    "build_quantum_classifier",
    "build_sampler_qnn",
]
