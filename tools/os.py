import os as os

__all__ = \
['listdir']

def listdir(path, isdir=True):
    if isdir:
        return [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
    else:
        return os.listdir(path)
