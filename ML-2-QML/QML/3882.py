import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

def _clip(value: float, bound: float) -> float:
    """Clip a real value to a symmetric range."""
    return max(-bound, min(bound, value))

class PhotonicFraudDetection:
    """Photonic fraud‑detection circuit with optional pre‑encoding of image statistics.

    The circuit implements the layered structure from the original fraud‑detection seed.
    Two image statistics (mean of left/right halves) are encoded into the displacement
    amplitudes of the two modes before the photonic layers.
    """

    def __init__(self, params: 'FraudLayerParameters') -> None:
        self.params = params

    def _apply_layer(self, modes: Sequence, params: 'FraudLayerParameters', *, clip: bool) -> None:
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else _clip(k, 1)) | modes[i]

    def encode_image(self, img: np.ndarray) -> sf.Program:
        """Encode a single‑channel image into a 2‑mode photonic program."""
        h, w = img.shape
        half = w // 2
        left_mean = img[:, :half].mean()
        right_mean = img[:, half:].mean()

        prog = sf.Program(2)
        with prog.context as q:
            Dgate(left_mean, 0) | q[0]
            Dgate(right_mean, 0) | q[1]
            self._apply_layer(q, self.params, clip=True)
        return prog

    def run(
        self,
        img: np.ndarray,
        backend: sf.backends.Backend,
        shots: int = 8192,
    ) -> sf.Result:
        """Execute the encoded program on the given backend."""
        prog = self.encode_image(img)
        return backend.run(prog, shots=shots)

__all__ = ["PhotonicFraudDetection"]
