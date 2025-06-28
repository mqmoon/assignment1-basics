

# CHUNKS = 64 #tunable



# def find_chunk_boundaries(
#     file: BinaryIO, 
#     desired_num_chunks: int, 
#     split_special_token: bytes
# ) -> list[int]:
#     """
#     Chunk the file into parts that can be counted independently.
#     May return fewer chunks if the boundaries end up overlapping.
#     """
#     assert isinstance(split_special_token, bytes), (
#         "Must represent special token as a bytestring"
#     )

#     # Get total file size in bytes
#     file.seek(0, os.SEEK_END)
#     file_size = file.tell()
#     file.seek(0)

#     chunk_size = file_size // desired_num_chunks

#     # Initial guesses for chunk boundary locations, uniformly spaced
#     # Chunks start on previous index, don't include last index
#     chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
#     chunk_boundaries[-1] = file_size

#     mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

#     for bi in range(1, len(chunk_boundaries) - 1):
#         initial_position = chunk_boundaries[bi]
#         file.seek(initial_position)  # Start at boundary guess
#         while True:
#             mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

#             # If EOF, this boundary should be at the end of the file
#             if mini_chunk == b"":
#                 chunk_boundaries[bi] = file_size
#                 break

#             # Find the special token in the mini chunk
#             found_at = mini_chunk.find(split_special_token)
#             if found_at != -1:
#                 chunk_boundaries[bi] = initial_position + found_at
#                 break
#             initial_position += mini_chunk_size

#     # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
#     return sorted(set(chunk_boundaries))


# def pretokenize(file_path: str, start: int, end: int):
#     try:
#         with open(file_path, "rb") as f:
#             # print(f.tell())
#             # print(f'start: {start}, end: {end}\n')
#             f.seek(start)
#             num_bytes_to_read = end - start
#             chunk_bytes = f.read(num_bytes_to_read)
#             chunk_texts = chunk_bytes.decode('utf-8')
#             # cleaned_text = chunk_texts.replace('\r\n', '\n').replace('\r', '')
#             sub_chunks = re.split(re.escape("<|endoftext|>"), chunk_texts)
            
#             PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
#             pretokenize_results = []
#             for chunk in sub_chunks:
#                 match_iter = re.finditer(PAT, chunk)
#                 pretokenize_result = [match.group(0) for match in match_iter]
#                 pretokenize_results.extend(pretokenize_result)
#             return pretokenize_results
#     except Exception as e:
#         print(f'pretokenize出错!{e}')
#         return []
    




# def pretokenize_and_count_chunk(text: str):
#     try:
#         for m in re.finditer(PAT, text):
#             word = m.group(0)
#             symbol_freqs[to_bytes_tuple(word)] += 1
#     except Exception as e:
#         print(f'Error occured in pretokenize_chunk()')
        # return []


# def get_word_freqs(list_of_token_lists: list[list[str]]) -> dict[str, int]:
#     word_freqs = defaultdict(int)
#     for token_list in list_of_token_lists:
#         for token in token_list:
#             word_freqs[token] += 1
#     return word_freqs


# def get_pair_stats(splits: dict[str, list[int]], word_freqs: dict[str, int]) -> defaultdict[tuple[int, int], int]:
#     pair_stats = defaultdict(int)
#     for word, freq in word_freqs.items():
#         symbols = splits[word]
#         for i in range(len(symbols) - 1):
#             pair_stats[(symbols[i], symbols[i+1])] += freq
#     return pair_stats




# def merge_pair(a: int, b: int, splits: dict[str, list[int]], new_id: int) -> dict[str, list[int]]:
#     """在语料库中执行整数对的合并。"""
#     new_splits = {}
#     for word, symbols in splits.items():
#         new_symbols = []
#         i = 0
#         while i < len(symbols):
#             if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
#                 new_symbols.append(new_id)
#                 i += 2
#             else:
#                 new_symbols.append(symbols[i])
#                 i += 1
#         new_splits[word] = new_symbols
#     return new_splits

import time
import multiprocessing
import os
import regex as re
import pickle
from collections import defaultdict
from typing import BinaryIO, Tuple, DefaultDict, Optional

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_pair_stats(symbol_freqs, pair_stats):
    for symbol, freq in symbol_freqs.items():
        for i in range(len(symbol) - 1):
            pair_stats[(symbol[i], symbol[i+1])] += freq


def to_bytes_tuple(word: str) -> Tuple[bytes]:
    l = list(tuple(word.encode("utf-8")))
    l = [bytes([x]) for x in l]
    return tuple(l)


def train_bpe(input_path, vocab_size: int, 
              special_tokens: list[str], prefix: Optional[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        assert vocab_size >= 256 + len(special_tokens)
        print(f'starting to train bpe with input_path: {input_path}, vocab_size: {vocab_size}, special_tokens: {special_tokens}, prefix: {prefix}')
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
        save_tokenizer_pickle(vocab=vocab, merges=merges, prefix=prefix)
        return vocab, merges


def save_tokenizer_pickle(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], prefix: str):
    vocab_file = f"{prefix}_vocab.pkl"
    merges_file = f"{prefix}_merges.pkl"

    print(f"正在使用 pickle 保存词汇表到 {vocab_file}...")
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)

    print(f"正在使用 pickle 保存合并规则到 {merges_file}...")
    with open(merges_file, 'wb') as f:
        pickle.dump(merges, f)
    
    print("保存完成。")


def load_tokenizer_pickle(prefix: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """使用 pickle 加载 vocab 和 merges"""
    vocab_file = f"{prefix}_vocab.pkl"
    merges_file = f"{prefix}_merges.pkl"

    print(f"\n正在从 {vocab_file} 加载词汇表...")
    # 使用 'rb' 模式进行二进制读取
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)

    print(f"正在从 {merges_file} 加载合并规则...")
    with open(merges_file, 'rb') as f:
        merges = pickle.load(f)
        
    print("加载完成。")
    return vocab, merges


if __name__ == "__main__":
    start_time = time.perf_counter()
    prefix = 'owt_valid'
    # file_path = "../data/pretok_test.txt"
    file_path = "data/owt_valid.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    train_bpe(file_path, vocab_size, special_tokens, prefix)
    loaded_vocab, loaded_merges = load_tokenizer_pickle(prefix)
    print(f"\n加载的词汇表: {loaded_vocab}")
    print(f"加载的合并规则: {loaded_merges}")
    print(f'training stopped, time consumed: {time.perf_counter() - start_time} s.')