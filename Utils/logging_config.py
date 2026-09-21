import logging
import os
import sys


def setup_logging(filepath, console_level=logging.INFO, file_level=logging.DEBUG):
    
    directory = os.path.dirname(os.path.abspath(filepath))
    if directory:
        os.makedirs(directory, exist_ok=True)

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(min(console_level, file_level))

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(filepath, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # librerie di terze parti: a DEBUG riempiono il log di rumore
    for nome in ("urllib3", "botocore", "s3transfer", "git", "git.util"):
        logging.getLogger(nome).setLevel(logging.WARNING)

    return root
