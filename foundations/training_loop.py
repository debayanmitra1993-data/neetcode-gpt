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
        n_dim = X.shape[1]
        weights = np.zeros(n_dim)
        bias = 0
        for epoch in range(epochs):
            # compute predictions..
            y_pred = (X @ weights) + bias
            
            # compute gradients..
            grad_w, grad_b = self.compute_gradients(X, y_pred, y)

            # update weight..
            weights = weights - (lr * grad_w)
            bias = bias - (lr * grad_b)

        return np.round(weights, 5), np.round(bias, 5)

    
    def compute_gradients(self, X, y_pred, y):
        grad_w_sum = 0
        grad_b_sum = 0
        n_rows = X.shape[0]
        for rowidx in range(len(X)):
            x_point = X[rowidx]
            y_point = y[rowidx]
            y_pred_point = y_pred[rowidx]
            grad_w_point = 2 * (y_pred_point - y_point) * (x_point)
            grad_b_point = 2 * (y_pred_point - y_point)
            grad_w_sum += grad_w_point
            grad_b_sum += grad_b_point
        grad_w = grad_w_sum / n_rows
        grad_b = grad_b_sum / n_rows
        return grad_w, grad_b
