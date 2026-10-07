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
        z = np.dot(x, w) + b
        y_hat = self.sigmoid(z)
        
        grad_w = np.zeros(x.shape[0])
        grad_b = np.round((y_hat - y_true)*y_hat*(1 - y_hat), 5)

        for dimidx in range(len(x)):
            grad_w[dimidx] = (y_hat - y_true)*y_hat*(1 - y_hat)*x[dimidx]
        
        return np.round(grad_w, 5), grad_b
        

        

