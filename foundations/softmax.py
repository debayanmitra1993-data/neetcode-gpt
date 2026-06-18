import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maxval = np.max(z)
        z = z - maxval
        sumval = np.sum(np.exp(z))
        print("sumval = ", sumval)

        return np.round(np.exp(z) / sumval,4)
