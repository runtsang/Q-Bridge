import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

class ConvDual:
    """
    Quantum convolutional filter for quanvolution layers.  The filter
    is a parameterized variational circuit of depth `n_layers`.  Each
    input patch of shape (kernel_size, kernel_size) is encoded as
    RX rotations on the qubits.  The circuit then applies `n_layers`
    of trainable RY rotations followed by CNOT entanglement.  The
    output is the expectation value of PauliZ on the first qubit,
    which is then averaged over all patches in a batch.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 n_layers: int = 2,
                 shots: int = 1024,
                 backend: str | None = None):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_layers = n_layers
        self.shots = shots
        self.backend = backend or Aer.get_backend('qasm_simulator')
        self.n_qubits = kernel_size ** 2

        # Parameter vector for encoding
        self.enc_params = ParameterVector('enc', length=self.n_qubits)

        # Parameter vector for variational layers
        self.var_params = ParameterVector('var', length=n_layers * self.n_qubits)

        # Build circuit
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        qc = QuantumCircuit(self.n_qubits)
        # Input encoding
        for i in range(self.n_qubits):
            qc.rx(self.enc_params[i], i)
        qc.barrier()
        # Variational layers
        for l in range(self.n_layers):
            for i in range(self.n_qubits):
                qc.ry(self.var_params[l * self.n_qubits + i], i)
            # Entanglement
            for i in range(0, self.n_qubits - 1, 2):
                qc.cx(i, i + 1)
        qc.measure_all()
        return qc

    def run(self, data):
        """
        Run the quantum circuit on a batch of image patches.

        Args:
            data: array of shape (batch, kernel_size, kernel_size)

        Returns:
            float: average expectation value of PauliZ on qubit 0 across
                   all patches.
        """
        batch = data.shape[0]
        exp_vals = []
        for patch in data:
            # Bind encoding parameters to pixel values
            param_binds = {self.enc_params[i]: np.pi if val > self.threshold else 0
                           for i, val in enumerate(patch.flatten())}
            # Bind variational parameters (using current values)
            param_binds.update({self.var_params[i]: 0.0 for i in range(len(self.var_params))})
            # Execute
            job = execute(self.circuit,
                          backend=self.backend,
                          shots=self.shots,
                          parameter_binds=[param_binds])
            result = job.result()
            counts = result.get_counts()
            # Compute expectation of PauliZ on qubit 0
            exp = 0
            for bitstring, n in counts.items():
                # bitstring order is reversed: qubit 0 is last
                bit0 = int(bitstring[-1])
                exp += (1 - 2 * bit0) * n
            exp /= self.shots
            exp_vals.append(exp)
        return np.mean(exp_vals)

__all__ = ["ConvDual"]
