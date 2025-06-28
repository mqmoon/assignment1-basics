import os
from typing import BinaryIO
import multiprocessing
import regex as re
import time

TEXT_FILE_PATH = '../data/TinyStoriesV2-GPT4-valid.txt'
# TEXT_FILE_PATH = '../data/pretok_test.txt'
CHUNKS = 128 # tunable

# parallelizable
def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size =4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(file_path: str, start: int, end: int):
    try:
        with open(file_path, "rb") as f:
            # print(f.tell())
            # print(f'start: {start}, end: {end}\n')
            f.seek(start)
            num_bytes_to_read = end - start
            chunk_bytes = f.read(num_bytes_to_read)
            chunk_texts = chunk_bytes.decode('utf-8')
            sub_chunks = re.split(re.escape("<|endoftext|>"), chunk_texts)
            
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            pretokenize_results = []
            for chunk in sub_chunks:
                match_iter = re.finditer(PAT, chunk)
                pretokenize_result = [match.group(0) for match in match_iter]
                pretokenize_results.extend(pretokenize_result)
            return pretokenize_results
    except Exception as e:
        print(f'pretokenize出错!{e}')
        return []


# if __name__ == "__main__":
#     ## Usage
#     start_time = time.perf_counter()
#     with open(TEXT_FILE_PATH, "rb") as f:
#         boundaries = find_chunk_boundaries(
#             f, CHUNKS, "<|endoftext|>".encode("utf-8"))
            
#         # The following is a serial implementation, but you can parallelize this 
#         # by sending each start/end pair to a set of processes.
#         # for start, end in zip(boundaries[:-1], boundaries[1:]):
#         #     f.seek(start)
#         #     chunk = f.read(end - start).decode("utf-8", errors="ignore")
#             # Run pre-tokenization on your chunk and store the counts for each pre-token
#         pretok_tasks = []
#         for i in range(len(boundaries)-1):
#             chunk_start = boundaries[i]
#             chunk_end = boundaries[i+1]
#             print(f'chunk_start: {chunk_start}, chunk_end: {chunk_end}')
#             pretok_tasks.append((TEXT_FILE_PATH, chunk_start, chunk_end))
        
#         # print(pretok_tasks)
        
#         with multiprocessing.Pool() as pool:
#             pre_tokenization_results = pool.starmap(pretokenize, pretok_tasks)
        
#         print(len(pre_tokenization_results))
#         print(type(pre_tokenization_results))
#         for i in range(len(pre_tokenization_results)):
#             if i == len(pre_tokenization_results) - 1:
#                 print(f'result {i}: {pre_tokenization_results[i]}\n\n\n')
#     print(f'time consuming: {time.perf_counter() - start_time} s.')