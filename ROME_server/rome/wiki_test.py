import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

wiki = load_dataset(
    "./wikipedia.py",
    cache_dir="./datasets",
    beam_runner="DirectRunner",
    language="en",
    date="20230701",
    # date="20230626",
)["train"]
print("test")
wiki[0]