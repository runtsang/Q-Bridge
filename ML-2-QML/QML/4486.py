from qiskit import QuantumCircuit, transpile, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
import networkx as nx
import qutip as qt
import numpy as np
from typing import List, Tuple

class QuantumClassifierModel:
    """
    Quantum hybrid classifier that builds a data‑uploading ansatz, samples with a
    Qiskit SamplerQNN, and constructs a fidelity graph of the final state.
    """
    def __init__(self, num_qubits: int, depth: int, num_classes: int = 2):
        self.num_qubits = num_qubits
        self.depth = depth
        self.num_classes = num_classes
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()
        # SamplerQNN from Qiskit Machine Learning
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.encoding,
            weight_params=self.weights,
            sampler=Aer.get_backend("qasm_simulator")
        )
        self.fidelity_graph: nx.Graph | None = None

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[qt.Qobj]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        qc = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        # Observables: Pauli‑Z on each qubit
        observables = [
            qt.tensor(*[qt.qeye(2) if k!= i else qt.sigmaz() for k in range(self.num_qubits)])
            for i in range(self.num_qubits)
        ]
        return qc, list(encoding), list(weights), observables

    def evaluate(self, inputs: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Run the circuit with given inputs and variational parameters, returning sampled probabilities.
        """
        param_dict = {p: v for p, v in zip(self.encoding, inputs)}
        param_dict.update({p: v for p, v in zip(self.weights, theta)})
        bound = self.circuit.bind_parameters(param_dict)
        # Use SamplerQNN to get samples
        samples = self.sampler_qnn.sample(inputs=inputs, weight_params=theta, shots=1024)
        # Convert to probability distribution
        probs = np.bincount(samples, minlength=2**self.num_qubits) / samples.size
        # Build fidelity graph of the final state
        statevec = Statevector.from_instruction(bound)
        state_qobj = qt.Qobj(statevec.data, dims=[[2]*self.num_qubits, [1]*self.num_qubits])
        self.fidelity_graph = self._build_fidelity_graph(state_qobj)
        return probs

    def _build_fidelity_graph(self, state: qt.Qobj) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_qubits))
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                # Reduced fidelity between qubits i and j
                rho_i = qt.partial_trace(state, [k for k in range(self.num_qubits) if k!= i])
                rho_j = qt.partial_trace(state, [k for k in range(self.num_qubits) if k!= j])
                fid = abs((rho_i.dag() * rho_j)[0, 0])  # approximate fidelity
                if fid > 0.8:
                    graph.add_edge(i, j, weight=float(fid))
        return graph

    def get_fidelity_graph(self) -> nx.Graph | None:
        """Return the last computed fidelity graph or None if not yet computed."""
        return self.fidelity_graph

__all__ = ["QuantumClassifierModel"]
