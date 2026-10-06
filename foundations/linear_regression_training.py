import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        
        iter_cnt = 0
        weights = initial_weights
        while iter_cnt < num_iterations:
            grad_w = np.zeros(len(weights))

            for rowidx in range(len(X)):
                x_pt = X[rowidx]
                pred = np.dot(x_pt, weights)
                error = Y[rowidx] - pred
                grad_w += (-2*error)*x_pt
            grad_w = np.round(grad_w / len(X), 5)

            weights = weights - (grad_w * Solution.learning_rate)

            iter_cnt += 1
        
        return np.round(weights, 5)






