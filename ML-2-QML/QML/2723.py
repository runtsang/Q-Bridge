from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit_aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler

class QuantumHybridSamplerAttention:
    """
    Quantum sampler that incorporates a parameter‑ized self‑attention style
    circuit. Rotation parameters are split into per‑qubit RX/RZ/RY gates,
    while entanglement parameters control a chain of controlled‑RX gates.
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        # Parameters that mirror the classical attention matrices
        self.rotation_params = ParameterVector("rot", 3 * n_qubits)
        self.entangle_params = ParameterVector("ent", n_qubits - 1)
        self.circuit = self._build_circuit()
        self.sampler = StatevectorSampler()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        # Per‑qubit rotations
        for i in range(self.n_qubits):
            qc.rx(self.rotation_params[3 * i], i)
            qc.ry(self.rotation_params[3 * i + 1], i)
            qc.rz(self.rotation_params[3 * i + 2], i)
        # Entangling chain
        for i in range(self.n_qubits - 1):
            qc.crx(self.entangle_params[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self, backend: AerSimulator, inputs: np.ndarray, shots: int = 1024) -> dict:
        """
        Encode the input vector into the first rotation parameter of each qubit
        and execute the circuit on the given backend.
        """
        if len(inputs)!= self.n_qubits:
            raise ValueError("Input length must match number of qubits.")
        # Map inputs to rotation parameters (only the RX component for simplicity)
        for i, val in enumerate(inputs):
            self.rotation_params[3 * i] = val
        job = execute(self.circuit, backend, shots=shots)
        return job.result().get_counts(self.circuit)

# Convenience wrapper that mimics Qiskit Machine Learning's SamplerQNN
def SamplerQNN():
    backend = AerSimulator()
    attention = QuantumHybridSamplerAttention(n_qubits=4)
    return QSamplerQNN(
        circuit=attention.circuit,
        input_params=attention.rotation_params,
        weight_params=attention.entangle_params,
        sampler=StatevectorSampler()
    )

__all__ = ["QuantumHybridSamplerAttention", "SamplerQNN"]
