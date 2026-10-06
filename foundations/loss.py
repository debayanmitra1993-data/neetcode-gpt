import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred_min_clip = np.array([1e-7]*len(y_pred))
        y_pred_max_clip = np.array([1 - 1e-7]*len(y_pred))

        y_pred = np.maximum(y_pred, y_pred_min_clip)
        y_pred = np.minimum(y_pred, y_pred_max_clip)

        sum_arr = (np.log(y_pred) * y_true) + ((1 - y_true)*np.log(1 - y_pred))
        return np.round(np.sum(sum_arr) * (-1/len(y_true)), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred_min_clip = np.array([[1e-7]*y_pred.shape[1]]*y_pred.shape[0])
        y_pred_max_clip = np.array([[1 - 1e-7]*y_pred.shape[1]]*y_pred.shape[0])

        y_pred = np.maximum(y_pred, y_pred_min_clip)
        y_pred = np.minimum(y_pred, y_pred_max_clip)

        actual_labels = y_true == 1
        probabilities_labels = np.log(y_pred[actual_labels])
        print("p_hat = ", probabilities_labels)
        return np.round(np.sum(probabilities_labels)*(-1/y_true.shape[0]),4)
        
