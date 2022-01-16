from typing import Iterable

from tqdm import tqdm


def progress_bar(iterable: Iterable, desc: str = "", num_cols: int = 80) -> Iterable:
    bar_format = "{percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    if len(desc) > 0:
        bar_format = "{desc}: " + bar_format
    return tqdm(iterable, desc=desc, bar_format=bar_format, ncols=num_cols, leave=False)
