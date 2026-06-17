import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        grad_w = [0]*len(w)
        y_pred = self.sigmoid(np.dot(w, x) + b)

        for dimidx in range(len(grad_w)):
            grad_w[dimidx] = round((y_pred - y_true)*y_pred*(1 - y_pred)*x[dimidx], 5)
        grad_b = round((y_pred - y_true)*y_pred*(1 - y_pred), 5)

        return (np.array(grad_w), grad_b)


