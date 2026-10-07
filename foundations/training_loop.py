import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        n_col = X.shape[1]
        n_row = X.shape[0]
        weights = np.zeros(n_col)
        bias = 0

        for epoch in range(epochs):

            grad_w = np.zeros(n_col)
            grad_b = 0

            for rowidx in range(n_row):
                x_pt = X[rowidx]
                y_pt = y[rowidx]
                y_pred_pt = np.dot(x_pt, weights) + bias

                grad_w += 2*(y_pred_pt - y_pt)*x_pt
                grad_b += 2*(y_pred_pt - y_pt)
            
            grad_w = grad_w/n_row
            grad_b = grad_b/n_row

            weights = weights - (grad_w * lr)
            bias = bias - (grad_b * lr)
        
        return (np.round(weights, 5), np.round(bias, 5))

