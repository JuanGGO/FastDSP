import os


def find_in_path(name, path):
    """
    Find a file in a search path
    """
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None