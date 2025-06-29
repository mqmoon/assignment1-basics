from typing import Iterable, Iterator
import regex as re
import pickle
import time

def get_pairs(ids: list[bytes]) -> set:
    """Helper function to find all adjacent pairs of tokens in a list."""
    pairs = set()
    for pair in zip(ids, ids[1:]):
        pairs.add(pair)
    return pairs

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.token_to_id = {v: k for k, v in self.vocab.items()}

        # Merges are more efficient as a dictionary mapping (b1, b2) -> rank
        self.merges = {pair: i for i, pair in enumerate(merges)}
        
        self.special_tokens = {}
        self.special_tokens_inverse = {}
        
        # A regex pattern to split text by special tokens, keeping them as delimiters
        self.special_token_pattern = ""
        if special_tokens:
            self._add_special_tokens(special_tokens)
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                   special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)

        # Load merges from a pickle file
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        final_token_ids = []
        # If there are special tokens, we first split the text by them.
        if self.special_token_pattern:
            text_chunks = re.split(self.special_token_pattern, text)
        else:
            text_chunks = [text]
        for chunk in text_chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                # If the chunk is a known special token, we add its ID directly.
                final_token_ids.append(self.special_tokens[chunk])
            else:
                # This is a normal text chunk, so we apply the BPE algorithm.
                # Step 1: Pre-tokenize the chunk into smaller parts (words, punctuation, etc.).
                pre_tokens = re.findall(self.pat, chunk)
                for pre_token in pre_tokens:
                    # Step 2: Convert the pre-token string into a list of single-byte `bytes` objects.
                    byte_sequence = [bytes([b]) for b in pre_token.encode('utf-8')]
                    if not byte_sequence:
                        continue
                    # Step 3: Iteratively apply merges.
                    while len(byte_sequence) > 1:
                        # Find all adjacent pairs of byte symbols in the current sequence.
                        pairs = get_pairs(byte_sequence)
                        # Find the pair with the highest priority (lowest rank in merges).
                        # If a pair is not in our merges list, it gets a rank of infinity.
                        best_pair = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
                        # If the best pair has a rank of infinity, it means no more merges are possible.
                        if best_pair not in self.merges:
                            break
                        # Merge all occurrences of the best pair in the sequence.
                        new_byte_sequence = []
                        i = 0
                        while i < len(byte_sequence):
                            if i < len(byte_sequence) - 1 and (byte_sequence[i], byte_sequence[i+1]) == best_pair:
                                # Merge the two bytes objects and append the result.
                                merged = byte_sequence[i] + byte_sequence[i+1]
                                new_byte_sequence.append(merged)
                                i += 2 # Move past the two merged bytes.
                            else:
                                new_byte_sequence.append(byte_sequence[i])
                                i += 1
                        byte_sequence = new_byte_sequence
                    # After the loop, 'byte_sequence' contains the final token bytes for this pre-token.
                    # Now, convert these final byte symbols to their corresponding integer IDs.
                    for token_bytes in byte_sequence:
                        token_id = self.token_to_id.get(token_bytes)
                        if token_id is not None:
                            final_token_ids.append(token_id)
                        else:
                            # Fallback for robustness: if a merged token is somehow not in the vocab,
                            # encode its constituent bytes individually. This assumes all single
                            # bytes (0-255) are in the initial vocabulary.
                            for byte_val in token_bytes:
                                final_token_ids.append(self.token_to_id[bytes([byte_val])])

        return final_token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def decode(self, ids: list[int]) -> str:
        # Collect all byte sequences corresponding to the token IDs
        byte_chunks = [self.vocab.get(token_id, b'') for token_id in ids]
        
        # Concatenate all byte chunks into a single bytes object
        full_byte_sequence = b"".join(byte_chunks)
        
        # Decode the full byte sequence into a string, replacing any malformed
        # UTF-8 sequences with the Unicode replacement character U+FFFD.
        return full_byte_sequence.decode('utf-8', errors='replace')
    
    def _add_special_tokens(self, special_tokens: list[str]):
        """Internal method to add special tokens to the vocabulary."""
        for token_str in special_tokens:
            # Check if the special token is already in our vocabulary
            # Special tokens are encoded directly as UTF-8 bytes
            token_bytes = token_str.encode('utf-8')
            if token_bytes in self.token_to_id:
                token_id = self.token_to_id[token_bytes]
            else:
                # If not, add it with a new ID
                token_id = len(self.vocab)
                self.vocab[token_id] = token_bytes
                self.token_to_id[token_bytes] = token_id

            self.special_tokens[token_str] = token_id
            self.special_tokens_inverse[token_id] = token_str
        # Sort special tokens by length in descending order. This is crucial.
        # It ensures that longer tokens (e.g., "<|eot|><|eot|>") are matched
        # before shorter, overlapping tokens (e.g., "<|eot|>") in the regex.
        sorted_tokens = sorted(self.special_tokens.keys(), key=len, reverse=True)
        escaped_tokens = [re.escape(t) for t in sorted_tokens]
        self.special_token_pattern = f"({ '|'.join(escaped_tokens) })"
        # # Create a regex to split the text by any of the special tokens
        # # The capturing group ( ... ) ensures the delimiters are kept in the result
        # escaped_tokens = [re.escape(t) for t in self.special_tokens]
        # self.special_token_pattern = f"({ '|'.join(escaped_tokens) })"
    
if __name__ == "__main__":
    prefix = 'owt_valid'
    vocab_file = f"{prefix}_vocab.pkl"
    merges_file = f"{prefix}_merges.pkl"
    special_tokens = ['<|endoftext|>']
    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_file, merges_filepath=merges_file, special_tokens=special_tokens)
    # print(len(tokenizer.vocab))
    # print(len(tokenizer.merges))
    # print(tokenizer.special_tokens)
    with open ('data/owt_valid.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    encode_results = tokenizer.encode(text=text)
    print(text)
    print(encode_results)
    print(tokenizer.decode(encode_results))