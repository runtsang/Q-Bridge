from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 1. Photonic‑style fraud layer (quantum version)
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """
    Assemble a Strawberry‑Fields‑style photonic circuit.
    For the quantum module we simply return a Qiskit equivalent that applies
    the same sequence of gates using the corresponding Qiskit primitives.
    """
    qc = QuantumCircuit(2)
    # Input layer (no clipping)
    qc.append(_apply_layer_gates, [0, 1], params=input_params, clip=False)
    for layer in layers:
        qc.append(_apply_layer_gates, [0, 1], params=layer, clip=True)
    return qc

def _apply_layer_gates(qc: QuantumCircuit, params: FraudLayerParameters, clip: bool) -> None:
    # Beam‑splitter (BS) equivalent
    theta = params.bs_theta
    phi = params.bs_phi
    # Using a generic two‑qubit rotation as a placeholder
    qc.ry(theta, 0)
    qc.rz(phi, 1)
    # Phase rotations
    for i, phase in enumerate(params.phases):
        qc.rz(phase, i)
    # Squeezing (simulated via RZ as a placeholder)
    for i, (r, p) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_val = _clip(r, 5) if clip else r
        qc.rz(r_val, i)
    # Displacement (simulated via RX)
    for i, (d, p) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        d_val = _clip(d, 5) if clip else d
        qc.rx(d_val, i)
    # Kerr (simulated via RZ)
    for i, k in enumerate(params.kerr):
        k_val = _clip(k, 1) if clip else k
        qc.rz(k_val, i)

# --------------------------------------------------------------------------- #
# 2. Classical classifier construction (quantum analogue)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Sequence[ParameterVector], Sequence[ParameterVector], Sequence[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.
    Mirrors the structure used in the classical build_classifier_circuit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    # Data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

# --------------------------------------------------------------------------- #
# 3. Hybrid quanvolution + graph QNN + classifier
# --------------------------------------------------------------------------- #
class HybridQuanvolutionGraphClassifier:
    """
    Quantum counterpart of HybridQuanvolutionGraphClassifier.
    Encodes 4‑quadrant image patches into qubits, applies a random unitary
    (acting as a graph‑based QNN), then runs the variational classifier.
    """
    def __init__(self, num_qubits: int = 4, classifier_depth: int = 2):
        self.num_qubits = num_qubits
        self.classifier_circuit, self.enc_params, self.cls_params, self.observables = build_classifier_circuit(
            num_qubits, classifier_depth
        )
        # Random unitary to emulate a graph‑based QNN layer
        self.graph_unitary = self._random_unitary(num_qubits)
        self.backend = AerSimulator()
        self.backend.set_options(method="statevector")

    def _random_unitary(self, n: int) -> np.ndarray:
        """Generate a Haar‑random unitary matrix of size 2ⁿ × 2ⁿ."""
        dim = 2 ** n
        rng = np.random.default_rng()
        mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        q, _ = np.linalg.qr(mat)
        return q

    def _encode_image(self, img: np.ndarray) -> np.ndarray:
        """
        Convert a 28×28 grayscale image into 4 values representing the mean
        intensity of each quadrant. These values are used as parameters for
        RX rotations.
        """
        h, w = img.shape
        mid_h, mid_w = h // 2, w // 2
        q1 = img[:mid_h, :mid_w].mean()
        q2 = img[:mid_h, mid_w:].mean()
        q3 = img[mid_h:, :mid_w].mean()
        q4 = img[mid_h:, mid_w:].mean()
        return np.array([q1, q2, q3, q4])

    def forward(self, images: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        images : np.ndarray
            Batch of images of shape (batch, 28, 28) with pixel values in [0, 1].

        Returns
        -------
        np.ndarray
            Log‑softmax logits of shape (batch, 2).
        """
        batch_size = images.shape[0]
        all_logits = []

        for img in images:
            # Encode image into qubit parameters
            params = self._encode_image(img)
            qc = QuantumCircuit(self.num_qubits)
            for qubit, val in enumerate(params):
                qc.rx(val, qubit)

            # Apply graph‑based random unitary
            qc.unitary(self.graph_unitary, qc.qubits)

            # Append classifier ansatz
            qc.compose(self.classifier_circuit, inplace=True)

            # Measure expectation values of Z on each qubit
            meas = []
            for obs in self.observables:
                meas.append(qc.measure_z())

            job = execute(qc, self.backend, shots=1024)
            result = job.result()
            counts = result.get_counts(qc)
            # Convert counts to expectation values
            exp_vals = [
                sum(1 if bit == '1' else -1 for bit in key) * freq / 1024
                for key, freq in counts.items()
            ]
            # Simple log‑softmax over first two values
            logits = np.log(np.exp(exp_vals[:2]) / np.sum(np.exp(exp_vals[:2])))
            all_logits.append(logits)

        return np.vstack(all_logits)

    def fidelity_adjacency(self, states: Sequence[np.ndarray], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """
        Build a weighted adjacency graph from state fidelities.
        Used for graph‑based analyses of the quantum states produced by the model.
        """
        import networkx as nx
        import itertools

        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = np.abs(np.vdot(s_i, s_j)) ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "build_classifier_circuit",
    "HybridQuanvolutionGraphClassifier",
]
