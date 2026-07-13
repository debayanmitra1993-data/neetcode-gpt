import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        vocab = set()
        longest_sent = 0

        for positivesentence in positive:
            words = positivesentence.split(" ")
            if len(words) > longest_sent:
                longest_sent = len(words)
            for word in words:
                if word not in vocab:
                    vocab.add(word)
        for negativesentence in negative:
            words = negativesentence.split(" ")
            if len(words) > longest_sent:
                longest_sent = len(words)
            for word in words:
                if word not in vocab:
                    vocab.add(word)
        
        word_to_idx = {}
        word_count = 0

        for idx, word in enumerate(sorted(vocab)):
            word_to_idx[word] = word_count + 1
            word_count += 1

        output = []
        for positivesentence in positive:
            words = positivesentence.split(" ")
            thisoutput = []
            for word in words:
                thisoutput.append(word_to_idx[word])
            if len(thisoutput) < longest_sent:
                thisoutput = thisoutput + [0]*(longest_sent - len(thisoutput))
            output.append(thisoutput)
        for negativesentence in negative:
            words = negativesentence.split(" ")
            thisoutput = []
            for word in words:
                thisoutput.append(word_to_idx[word])
            if len(thisoutput) < longest_sent:
                thisoutput = thisoutput + [0]*(longest_sent - len(thisoutput))
            output.append(thisoutput)
        return output