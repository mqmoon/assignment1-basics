import re
from collections import defaultdict
from typing import Tuple, DefaultDict

# 定义正则表达式
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 把一个词转换成 (bytes, bytes, ...) 的 tuple
def to_bytes_tuple(word: str) -> Tuple[bytes]:
    l = list(tuple(word.encode("utf-8")))
    l = [bytes([x]) for x in l]
    return tuple(l)


# 计算所有 pair 的统计频率
def get_pair_stats(symbol_freqs, pair_stats):
    for symbol, freq in symbol_freqs.items():
        for i in range(len(symbol) - 1):
            pair_stats[(symbol[i], symbol[i+1])] += freq

def train_bpe(input_path, vocab_size: int, 
              special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        assert vocab_size >= 256 + len(special_tokens)
        
        with open(input_path, "r") as f:
            text = f.read()
        chunks = re.split("|".join(map(re.escape, special_tokens)), text)
        global symbol_freqs
        symbol_freqs = defaultdict(int)

        for chunk in chunks:
            for m in re.finditer(PAT, chunk):
                word = m.group(0)
                symbol_freqs[to_bytes_tuple(word)] += 1


        vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        for token_str in special_tokens:
            special_token_bytes = token_str.encode('utf-8')
            if special_token_bytes not in vocab.values():
                vocab[next_id] = special_token_bytes
                next_id += 1

        pair_stats: DefaultDict[Tuple[bytes, bytes], int] = defaultdict(int)
        get_pair_stats(symbol_freqs, pair_stats=pair_stats)
        merges = []

        while len(vocab) < vocab_size:
            pair_counts = defaultdict(int)

            # Count all adjacent byte pairs
            for token, cnt in symbol_freqs.items():
                for i in range(len(token) - 1):
                    pair = (token[i], token[i + 1])
                    pair_counts[pair] += cnt

            if not pair_counts:
                break  # No more pairs to merge

            # Find the most frequent pair(s)
            max_count = max(pair_counts.values())
            candidates = [k for k, v in pair_counts.items() if v == max_count]
            best_pair = max(candidates)

            a, b = best_pair

            # Create new token
            new_token = a + b
            vocab[next_id] = new_token
            next_id += 1

            # Apply the merge to all pre-tokenized sequences
            # 收集变更
            changes = []
            for token, cnt in symbol_freqs.items():
                # Find all occurrences of the `best_pair` in `token`
                indices = [i for i in range(len(token) - 1) if token[i:i + 2] == best_pair]
                if indices:
                    # Replace each occurrence with `new_token`
                    new_pre_token = []
                    i = 0
                    while i < len(token):
                        if i in indices:
                            new_pre_token.append(new_token)
                            i += 2
                        else:
                            new_pre_token.append(token[i])
                            i += 1
                    new_pre_token = tuple(new_pre_token)
                    changes.append((token, new_pre_token, cnt))

            # 应用变更
            for old_token, new_pre_token, cnt in changes:
                symbol_freqs[new_pre_token] = symbol_freqs.get(new_pre_token, 0) + cnt
                del symbol_freqs[old_token]

            # Record the merge
            merges.append((a, b))
        # for i in range(num_merges):
        #     if not pair_stats:
        #         break
            
        #     best_pair = max(pair_stats, key=lambda k: pair_stats[k])

        #     p1_bytes = best_pair[0]
        #     p2_bytes = best_pair[1]
        #     merges.append((p1_bytes, p2_bytes))

        #     vocab[next_id] = p1_bytes + p2_bytes
        #     for symbols, freq in symbol_freqs.items():
        #         new_symbols = []
        #         j = 0
        #         did_merge = False
                
        #         while j < len(symbols):
        #             if j < len(symbols) - 1 and (symbols[j], symbols[j+1]) == best_pair:
        #                 if j > 0:
        #                     pair_stats[(symbols[j-1], best_pair[0])] -= freq
        #                 if j < len(symbols) - 2:
        #                     pair_stats[(best_pair[1], symbols[j+2])] -= freq
        #                 new_symbols.append(p1_bytes + p2_bytes)
        #                 if j > 0:
        #                     pair_stats[(symbols[j-1], p1_bytes + p2_bytes)] += freq
        #                 if j < len(symbols) - 2:
        #                     pair_stats[(p1_bytes + p2_bytes, symbols[j+2])] += freq
                        
        #                 j += 2
        #                 did_merge = True
        #             else:
        #                 new_symbols.append(symbols[j])
        #                 j += 1
            
        #         if did_merge:
        #             symbols = new_symbols
        #     next_id = 256 + i
        return vocab, merges