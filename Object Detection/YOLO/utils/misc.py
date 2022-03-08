import os
import shutil
from tqdm import tqdm


def copy_extension(src, dst, ext):
    for file in tqdm(os.listdir(src)):
        if ext in file:
            file_path = os.path.join(src, file)
            shutil.copy(file_path, f"{dst}/{file}")

