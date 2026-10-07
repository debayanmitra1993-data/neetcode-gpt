import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)
        self.activation = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.out = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        z = self.fc1(images) # (batch_size, 512)
        z = self.activation(z)
        z = self.drop(z)
        z = self.fc2(z)
        z = self.out(z)
        return z