import os


def print_file_exists(filename: str) -> None:
    print(f"? file exists: {filename}\n-> {os.path.exists(filename)}")

