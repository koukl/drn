import os


def ensure_dir(d):
    if os.path.exists(d) and os.path.isdir(d):
        return
    else:
        os.makedirs(d)
