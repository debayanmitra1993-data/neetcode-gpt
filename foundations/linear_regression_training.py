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
        
        weights = initial_weights
        n_dim = X.shape[1]
        n_data = X.shape[0]

        for _ in range(num_iterations):
            
            # compute predictions
            Y_pred = X @ weights

            # compute gradients 
            grad_w = [0]*n_dim
            for rowidx in range(len(X)):
                x = X[rowidx]
                y = Y[rowidx]
                y_pred = Y_pred[rowidx]
                grad_w += 2*(y_pred - y)*x
            grad_w = grad_w/n_data

            # update weights
            weights = weights - (self.learning_rate * grad_w)
        
        return np.round(weights, 5)



