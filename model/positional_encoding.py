import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        pe = [[0 for _ in range(d_model)] for _ in range(seq_len)]
        for position in range(seq_len):
            for idx in range(d_model // 2):
                pe[position][2*idx] = np.round(np.sin(position / (10000 ** ((2*idx) / d_model))), 5)
                pe[position][(2*idx) + 1] = np.round(np.cos(position / (10000 ** ((2*idx) / d_model))), 5)
        return pe