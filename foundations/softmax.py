import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z_norm = z - np.max(z)
        z_exp = np.exp(z_norm)
        z_exp_sum = np.sum(z_exp)
        return np.round(z_exp/z_exp_sum, 4)
