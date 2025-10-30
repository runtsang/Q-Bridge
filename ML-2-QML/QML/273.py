import pennylane as qml
import pennylane.numpy as np

class ConvGen:
    """
    Variational quantum convolution filter that learns a kernel via gradient‑based
    back‑propagation on a hybrid simulator. The circuit encodes a 2×2 patch as
    rotation angles and applies a parameter‑shifted ansatz. The output is the
    average expectation value of Z on all qubits, optionally thresholded.
    """
    def __init__(self, kernel_size=2, threshold=0.5, device=None):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        if device is None:
            device = qml.device("default.qubit", wires=self.n_qubits, shots=1000)
        self.dev = device
        self.params = np.random.randn(self.n_qubits, 3)  # 3 parameters per qubit
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, data, params):
        for i in range(self.n_qubits):
            qml.RY(data[i], wires=i)
        for i in range(self.n_qubits):
            qml.RZ(params[i, 0], wires=i)
            qml.RX(params[i, 1], wires=i)
            qml.RZ(params[i, 2], wires=i)
            if i < self.n_qubits - 1:
                qml.CNOT(wires=[i, i+1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def run(self, data):
        """
        Run the quantum circuit on a 2D patch.
        Args:
            data: 2D array with shape (kernel_size, kernel_size).
        Returns:
            float: average expectation value of Z across qubits, thresholded.
        """
        data_flat = np.reshape(data, self.n_qubits)
        data_norm = np.pi * (data_flat - data_flat.min()) / (data_flat.max() - data_flat.min() + 1e-8)
        out = self.qnode(data_norm, self.params)
        mean_z = np.mean(out)
        return float(mean_z) if mean_z > self.threshold else float(0.0)

    def get_params(self):
        return self.params

    def set_params(self, new_params):
        self.params = new_params
