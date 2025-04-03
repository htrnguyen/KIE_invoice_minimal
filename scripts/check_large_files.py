import os
from pathlib import Path


def get_size_format(b, factor=1024, suffix="B"):
    """
    Scale bytes to its proper byte format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return f"{b:.2f}{unit}{suffix}"
        b /= factor
    return f"{b:.2f}Y{suffix}"


def find_large_files(directory=".", min_size=1024 * 1024):  # min_size = 1MB
    large_files = []

    for root, dirs, files in os.walk(directory):
        # Skip .git directory
        if ".git" in dirs:
            dirs.remove(".git")

        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_size = os.path.getsize(file_path)
                if file_size >= min_size:
                    large_files.append((file_path, file_size))
            except OSError:
                continue

    # Sort by size in descending order
    large_files.sort(key=lambda x: x[1], reverse=True)

    print("\nLarge files found (>= 1MB):")
    print("-" * 80)
    print(f"{'Size':>10} {'Path':<70}")
    print("-" * 80)

    for file_path, size in large_files:
        print(f"{get_size_format(size):>10} {file_path:<70}")


if __name__ == "__main__":
    find_large_files()
