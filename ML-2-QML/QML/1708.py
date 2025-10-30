"""Quantum convolutional filter with parameter‑shift gradient and adaptive threshold.

The ConvEnhanced class constructs a variational quantum circuit for a
kernel of size `kernel_size`.  Each input patch is encoded into a set
of RX rotations (π if the pixel value exceeds `threshold`, otherwise 0).
After encoding, a trainable Ry layer with parameters `theta` is applied,
followed by a linear RealAmplitudes entangling block.  The `run(data)`
method evaluates the average probability of measuring |1> across all
qubits.  The `gradient(data)` method computes the gradient of the
output with respect to the trainable parameters using the
parameter‑shift rule, enabling integration with classical optimizers.
"""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes

class ConvEnhanced:
    """
    Quantum filter for 2‑D convolution with trainable parameters.
    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel (n_qubits = kernel_size**2).
    threshold : float, default 0.0
        Threshold used for binary data encoding.
    shots : int, default 1024
        Number of shots for the simulator.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 shots: int = 1024):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        # Trainable parameters (one per qubit)
        self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        # Current parameter values (float)
        self.theta_vals = {p: 0.0 for p in self.theta}

    def _build_circuit(self, data_bit):
        """
        Internal helper that builds a circuit for a single data value.
        Parameters
        ----------
        data_bit : float
            Raw pixel value to be encoded.
        Returns
        -------
        QuantumCircuit
            Circuit ready for execution.
        """
        c = QuantumCircuit(self.n_qubits)
        # Data encoding: RX(pi) if pixel > threshold, else RX(0)
        angle = np.pi if data_bit > self.threshold else 0.0
        for i in range(self.n_qubits):
            c.rx(angle, i)
        # Trainable Ry layer
        for i, theta in enumerate(self.theta):
            c.ry(self.theta_vals[theta], i)
        # Entangling block (linear RealAmplitudes)
        c.barrier()
        c.append(RealAmplitudes(self.n_qubits, reps=1,
                                entanglement="linear"), range(self.n_qubits))
        c.measure_all()
        return c

    def run(self, data):
        """
        Evaluate the filter on a 2‑D patch.
        Parameters
        ----------
        data : np.ndarray
            Input patch of shape (kernel_size, kernel_size).
        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data_flat = data.reshape(-1)
        probs = []
        for val in data_flat:
            circ = self._build_circuit(val)
            job = execute(circ,
                          self.backend,
                          shots=self.shots)
            result = job.result()
            counts = result.get_counts(circ)
            total_ones = 0
            total_counts = 0
            for bitstring, freq in counts.items():
                total_ones += bitstring.count("1") * freq
                total_counts += freq
            probs.append(total_ones / (total_counts * self.n_qubits))
        return np.mean(probs)

    def gradient(self, data):
        """
        Compute the gradient of the output w.r.t. trainable parameters
        using the parameter‑shift rule.
        Parameters
        ----------
        data : np.ndarray
            Input patch of shape (kernel_size, kernel_size).
        Returns
        -------
        dict
            Mapping from Parameter to gradient value.
        """
        data_flat = data.reshape(-1)
        grads = {p: 0.0 for p in self.theta}
        for val in data_flat:
            # Evaluate expectation at +π/2 and –π/2 for each parameter
            for i, theta in enumerate(self.theta):
                orig_val = self.theta_vals[theta]
                # + shift
                self.theta_vals[theta] = np.pi / 2
                circ_plus = self._build_circuit(val)
                job_plus = execute(circ_plus,
                                   self.backend,
                                   shots=self.shots)
                res_plus = job_plus.result()
                counts_plus = res_plus.get_counts(circ_plus)
                prob_plus = 0.0
                for bitstring, freq in counts_plus.items():
                    prob_plus += bitstring.count("1") * freq
                prob_plus /= (self.shots * self.n_qubits)

                # - shift
                self.theta_vals[theta] = -np.pi / 2
                circ_minus = self._build_circuit(val)
                job_minus = execute(circ_minus,
                                    self.backend,
                                    shots=self.shots)
                res_minus = job_minus.result()
                counts_minus = res_minus.get_counts(circ_minus)
                prob_minus = 0.0
                for bitstring, freq in counts_minus.items():
                    prob_minus += bitstring.count("1") * freq
                prob_minus /= (self.shots * self.n_qubits)

                # restore original value
                self.theta_vals[theta] = orig_val

                grads[theta] += (prob_plus - prob_minus) / 2.0
        # Average over all data points
        for theta in grads:
            grads[theta] /= len(data_flat)
        return grads

    def parameters(self):
        """
        Return the list of trainable parameters for external optimizers.
        """
        return self.theta

    def set_parameters(self, values):
        """
        Update the trainable parameter values.
        Parameters
        ----------
        values : list or array of floats
            New values for the parameters in the same order as returned by
            `parameters()`.
        """
        for p, val in zip(self.theta, values):
            self.theta_vals[p] = val

    def set_threshold(self, th):
        """
        Update the encoding threshold.
        """
        self.threshold = th

__all__ = ["ConvEnhanced"]
