import os
def clean_csv(filename_path):
    with filename_path:
        os.remove(filename_path)
