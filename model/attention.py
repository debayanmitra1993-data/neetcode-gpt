import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.attention_dim = attention_dim
        self.K_matrix = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.Q_matrix = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.V_matrix = nn.Linear(embedding_dim, attention_dim, bias = False)


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        k = self.K_matrix(embedded)
        q = self.Q_matrix(embedded)
        v = self.V_matrix(embedded)

        att_scores = q @ torch.transpose(k, 1, 2) 
        att_scores = att_scores / math.sqrt(self.attention_dim)
        
        causal_mask = torch.tril(att_scores) == 0
        att_scores[causal_mask] = float("-inf")
        att_scores = torch.nn.functional.softmax(att_scores, dim = 2)

        return torch.round(att_scores @ v, decimals = 4)
