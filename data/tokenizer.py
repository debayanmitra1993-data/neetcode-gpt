from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        merges = []

        # Initialize character IDs
        itos = {ord(char): char for char in corpus}
        corpus_lst = [ord(char) for char in corpus]

        next_token_id = max(itos, default=-1) + 1

        count_merges = 0

        while count_merges < num_merges:

            # Count adjacent pairs
            corpus_store = {}

            for idx in range(len(corpus_lst) - 1):
                pair = (corpus_lst[idx], corpus_lst[idx + 1])
                corpus_store[pair] = corpus_store.get(pair, 0) + 1

            # Stop if no pairs remain
            if not corpus_store:
                break

            # Highest frequency first; break ties by token IDs
            best_pair = min(
                corpus_store,
                key=lambda pair: (-corpus_store[pair], pair)
            )

            best_pair_val_1, best_pair_val_2 = best_pair

            best_pair_chr_1 = itos[best_pair_val_1]
            best_pair_chr_2 = itos[best_pair_val_2]

            merged_pair = best_pair_chr_1 + best_pair_chr_2
            merged_pair_val = next_token_id
            next_token_id += 1

            # Record merge rule in order
            merges.append([best_pair_chr_1, best_pair_chr_2])

            itos[merged_pair_val] = merged_pair

            # Replace non-overlapping occurrences
            new_corpus_lst = []
            idx = 0

            while idx < len(corpus_lst):
                if (
                    idx + 1 < len(corpus_lst)
                    and corpus_lst[idx] == best_pair_val_1
                    and corpus_lst[idx + 1] == best_pair_val_2
                ):
                    new_corpus_lst.append(merged_pair_val)
                    idx += 2
                else:
                    new_corpus_lst.append(corpus_lst[idx])
                    idx += 1

            corpus_lst = new_corpus_lst
            count_merges += 1

        return merges