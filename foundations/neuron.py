import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        
        pre_activation = np.dot(x, w) + b
        return round(float(self.activation_func(activation, pre_activation)),5)
    
    def activation_func(self, activation, inputval):
        if activation == "sigmoid":
            return 1/(1 + np.exp(-inputval))
        elif activation == "relu":
            return max(0, inputval)


