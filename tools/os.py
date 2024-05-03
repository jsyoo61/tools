import os
from typing import Union
__all__ = \
[
'listdir',
'makedirs'
]

def makedirs(path, exist_ok=True):
    os.makedirs(path, exist_ok=exist_ok)

def listdir(path=None, isdir: bool=False, isfile: Union[bool, str]=False, join: bool=False):
    '''
    :func listdir:

    :param path:
    :param isdir:
    :param isfile:
    to get files which start with ".", set isfile=''.
    :param join:
    '''
    # Safety check
    if type(isfile)==str:
        ext = isfile.lower() # extension
        isfile_str = True
        isfile = True
    else:
        isfile_str=False
    # assert type(isfile)==str or type(isfile)==bool, 'isfile can be either str or bool'
    # _isfile = True if type(isfile)==str else isfile
    assert not (isdir and isfile), 'only one of argument "isdir" and "isfile" can be True'
    # if path == None:
    #     path = '.'

    dir_list = os.listdir(path)

    # when path is referring to somewhere non-cwd, we need dir_list_joined to pass reference for isdir() or isfile().
    if join or isdir or isfile:
        dir_list_joined = dir_list if path is None else [os.path.join(path, dir) for dir in dir_list]

    if join:
        dir_list = dir_list_joined

    if isdir:
        return [dir for dir, dir_joined in zip(dir_list, dir_list_joined) if os.path.isdir(dir_joined)]
    elif isfile:
        if isfile_str:
            return [dir for dir, dir_joined in zip(dir_list, dir_list_joined) if os.path.isfile(dir_joined) and (os.path.splitext(dir)[1].lower() == ext)]
        else:
            return [dir for dir, dir_joined in zip(dir_list, dir_list_joined) if os.path.isfile(dir_joined)]
    else:
        return dir_list

# if __name__ == '__main__':
#     listdir('torch', isdir=False, join=False)
#     listdir('torch', isdir=False, join=True)
#     listdir('torch', isdir=True, join=False)
#     listdir('torch', isdir=True, join=True)
#
#     listdir('torch', isfile=True, join=False)
#     listdir('torch', isfile=True, join=True)
#     listdir('torch', isfile='.p', join=False)
#     listdir('torch', isfile='.py', join=True)
#     listdir('torch', isfile='', join=True)
#
#     listdir('torch', isdir=True, isfile=True)
#     listdir('torch', isdir=True, isfile='py')
