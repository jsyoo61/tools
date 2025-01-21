from copy import deepcopy as dcopy
import itertools as it
import os
import warnings
import pathlib
import pickle
import subprocess
import time

from . import os as _os

__all__ = \
[
#  'Filename',
'Printer',
'TDict',
'Timer',
 'Path',
'Wrapper',
'append',
'cmd',
'equal',
'equal_set',
'equal_array',
'iseven',
'isodd',
'is_iterable',
'isnumeric',
'isint',
'load_pickle',
'now',
'reverse_dict',
'unnest_dict',
'nestdict_to_list',
'update_ld',
'update_keys',
'merge_dict',
'merge_tuple',
'prettify_dict',
'read',
'readline',
'readlines',
'save_pickle',
'str2bool',
'write'
]

 # def save_pickle(obj, path = None, protocol = None): # Typings
def save_pickle(obj: str, path: str = None, protocol: int = None):
    '''Save object as Pickle file to designated path.
    If path is not given, default to "YearMonthDay_HourMinuteSecond.p" '''
    if path == None:
        path = time.strftime('%y%m%d_%H%M%S.p')
        warnings.warn(f'Be sure to specify specify argument "path"!, saving as {path}...')
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)

def load_pickle(path: str):
    '''Load Pickle file from designated path'''
    with open(path, 'rb') as f:
        return pickle.load(f)

def write(content: str, path: str, encoding: str = None):
    with open(path, 'w', encoding = encoding) as f:
        f.write(content)

def append(content: str, path: str , encoding: str = None):
    with open(path, 'a', encoding = encoding) as f:
        f.write(content)

def read(path, encoding = None):
    with open(path, 'r', encoding = encoding) as f:
        text = f.read()
    return text

def readline(path, encoding = None):
    '''Create generator object which iterates over the file'''
    with open(path, 'r', encoding = encoding) as f:
        for line in f:
            yield line

def readlines(path, encoding = None):
    with open(path, 'r', encoding = encoding) as f:
        text = f.readlines()
    return text

def cmd(command: str, shell: bool = False, encoding: str = None):
    '''Run shell command and return stdout'''
    pipe = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding=encoding, shell=shell, text=True)
    return pipe.stdout

def equal(lst):
    '''return True if all elements in iterable is equal'''
    lst_inst = iter(lst)
    try:
        val = next(lst_inst)
    except StopIteration:
        return True

    for v in lst_inst:
        if v!=val:
            return False
    return True

def equal_set(*args):
    '''return True if each element in args has the same set content'''
    return equal([set(elem) for elem in args])

def equal_array(lst):
    '''return True if all elements in array are equal'''
    lst_inst = iter(lst)
    try:
        val = next(lst_inst)
    except StopIteration:
        return True

    for arr in lst_inst:
        if (arr!=val).any():
            return False
    return True

def iseven(i):
    return i%2==0

def isodd(i):
    return i%2==1

def reverse_dict(d):
    '''
    Reverse key:value pair of dict into value:key.
    Thus, values in dict must be hashable
    '''
    d_ = {}
    for k, v in d.items():
        d_[v]=k
    return d_

def unnest_dict(d, format='.'):
    '''
    Unnest nested dictionary.
    Nested keys are concatenated with "format"(defaults with ".") notation.

    ex)
    >>> rating = {'witcher': {'bombs': 5, 'swords': 7}, 'gta': {'guns': 5}, 'lol': {'magic': 5, 'skills': 3}}
    >>> unnest_dict(rating, format='.')
    {
    'witcher.bombs': 5,
    'witcher.swords': 7,
    'gta.guns': 5,
    'lol.magic': 5,
    'lol.skills': 3
    }
    '''
    un_d = {}
    # d = d.copy()
    for k, v in d.items():
        if type(v)==dict:
            un_d_ = unnest_dict(v, format=format)
            for k_, v_ in un_d_.items():
                un_d[str(k)+format+str(k_)] = v_
        else:
            un_d[k] = v
    return un_d

def nestdict_to_list(d, key=None):
    '''
    open 2nd order nested dict to list of dicts.
    If key is given, then key of the 1st order dict will be merged to the dicts as "key: 1st_key"
    If key is None, then the key of 1st order dict will be discarded.
    '''
    l = []
    for k, v in d.items():
        assert type(v) == dict
        d_ = {}
        if key != None:
            d_[key] = k
        d_.update(v)
        l.append(d_)

    return l

def update_ld(ld, d):
    '''add (key, value) to list of dicts'''
    ld = dcopy(ld)
    for d_ in ld:
        d_.update(d)

    return ld

def update_keys(d, d_key, copy=True): # inplace? copy?
    if copy:
        d = dcopy(d)
    for k_old, k_new in d_key.items():
        d[k_new] = d.pop(k_old)
    return d

def merge_dict(ld):
    '''merge list of dicts
    into dict of lists
    ld: list of dicts'''

    # keys = sorted(list(set([d.keys() for d in ld])))
    keys = sorted(list(set(it.chain(*[d.keys() for d in ld]))))
    d_merged = {key:[] for key in keys}
    for d in ld:
        for k, v in d.items():
            d_merged[k].append(v)
    return d_merged

def merge_tuple(lt):
    '''merge list of tuples
    into tuple of lists
    lt: list of tuples'''
    n_tuple = len(lt[0])
    l_merged = [[] for _ in range(n_tuple)]
    for t in lt:
        for i, v in enumerate(t):
            l_merged[i].append(v)
    return tuple(l_merged)

# def prettify_dict(d, print_type='yaml', **kwargs):
#     '''prettify long dictionary
#     Example
#     -------
#     >>> d = {'a':1, 'b':2, 'c':3}
#     >>> print_dict(d)
#
#     '''
#     if print_type == 'yaml':
#         return yaml.dump(d, **kwargs)
#     elif print_type=='pprint':
#         pp = pprint.PrettyPrinter()
#         return pp.pformat(info_basic)
#     else:
#         pass
def prettify_dict(dictionary, indent=0):
    return '\n'.join([' '*indent + str(k) +': '+str(v) if type(v)!=dict else str(k)+':\n'+prettify_dict(v, indent=indent+2) for k, v in dictionary.items()])

# def pipeline(functions, args):
#     from functools import reduce
#     return reduce(lambda acc, func: func(acc), functions, args)

# Used in Argparse
def str2bool(x):
    try:
        return bool(x)
    except:
        pass

    true_list = ['t', 'true', 'y', 'yes', '1']
    false_list = ['f', 'false', 'n', 'no', '0']
    if x.lower() in true_list:
        return True
    elif x.lower() in false_list:
        return False
    else:
        raise Exception('input has to be in one of two forms:\nTrue: %s\nFalse: %s'%(true_list, false_list))

# DEPRECATED: use str.isnumeric()
# def strisfloat(x):
#     try:
#         x=float(x)
#         return True
#     except ValueError:
#         return False

def isnumeric(x):
    try:
        float(x)
        return True
    except ValueError or TypeError:
        return False

def isint(x):
    try:
        int(x)
        return True
    except ValueError or TypeError:
        return False

def is_iterable(x):
    return hasattr(x, '__iter__')

def now(format: str ='-'):
    if format=='-':
        return time.strftime('%Y-%m-%d_%H-%M-%S')
    elif format=='_':
        return time.strftime('%y%m%d_%H%M%S')
    else:
        raise Exception("format has to be one of ['-', '_']")

# DEPRECATED: use pathlib.Path() instead
# class Filename():
#     '''
#     Class to handle Filename with suffix.
#     To call filename with suffix, call the instance.

#     Example
#     -------
#     >>> file = Filename('model', '.pt')
#     >>> file
#     (Filename, name: "model", suffix: ".pt")
#     >>> str(file)
#     "model"
#     >>> file()
#     "model.pt"
#     >>> (file+'1')
#     "model1"
#     >>> (file+'1')()
#     "model1.pt"
#     '''
#     def __init__(self,obj=None, suffix=''):
#         self.name = str(obj)
#         self.suffix = suffix

#     def __call__(self):
#         '''Add suffix and return str'''
#         return self.name+self.suffix

#     def __radd__(self, other):
#         return Filename(other+self.name,suffix=self.suffix)

#     def __add__(self, other):
#         return Filename(self.name+other,suffix=self.suffix)

#     def __mul__(self, other):
#         return Filename(self.name*other,suffix=self.suffix)

#     def __repr__(self):
#         return '(Filename, name: "%s", suffix: "%s")'%(self.name, self.suffix)

#     def __str__(self):
#         return self.name

# DEPRECATED: use pathlib.Path() instead

class Path(pathlib.Path):
    '''
    Joins paths by . syntax
    (Want to use pathlib.Path internally, but currently inherit from str)

    Parameters
    ----------
    path: str (default: '.')
        Notes the default path. Leave for default blank value which means the current working directory.
        So YOU MUST NOT USE "path" AS ATTRIBUTE NAME, WHICH WILL MESS UP EVERYTHING

    Example
    -------
    >>> path = Path('C:/exp')
    >>> path
    path: C:/exp

    >>> path.DATA = 'CelebA'
    >>> path
    path: C:/exp
    DATA: C:/exp/CelebA

    >>> path.PROCESSED = 'processed'
    >>> path.PROCESSED.M1 = 'method1'
    >>> path.PROCESSED.M2 = 'method2'
    >>> path
    path: C:/exp
    DATA: C:/exp/CelebA
    PROCESSED: C:/exp/processed

    >>> path.PROCESSED
    M1: C:/exp/processed/method1
    M2: C:/exp/processed/method2
    -------

    '''
    def __new__(cls, *args):
        if cls is Path:
            cls = WindowsPath2 if os.name == 'nt' else PosixPath2
        self = super().__new__(cls, *args)

        # self._path=Path(*args)
        # self = object.__new__(cls)
        return self
    
    # def __init__(self, *args):

    def __repr__(self):
        return f'tools.Path({super().__str__()})'

    def summary(self, indent=0):
        '''Print out current path, and children'''
        for name, _path in self.__dict__.items():
            print(' '*indent+name+': '+str(_path))
            if issubclass(type(_path), Path):
                _path.summary(indent+2)

            # if name != 'path':
        # print('\n'.join([key+': '+str(value) for key, value in self.__dict__.items()]))

    def __fspath__(self):
        return super().__fspath__()
        # return self._path.__fspath__()

    def __str__(self):
        return super().__str__()
        # return str(self._path)

    def __setattr__(self, key, value):
        # super(Path, self).__setattr__(key, self / value) # self.joinpath(value)
        if key.startswith('_'):
            super(Path, self).__setattr__(key, value)
        elif hasattr(self, key) and hasattr(getattr(self, key), '__call__'):
            raise AttributeError(f'Attribute "{key}" already exists')
        else:
            super(Path, self).__setattr__(key, Path(self / value))
        # if hasattr(self, 'path'):
        #     assert key != 'path', '"path" is a predefined attribute and must not be used. Use some other attribute name'
        #     super(Path, self).__setattr__(key, Path(os.path.join(self._path, value)))
        # else:
        #     super(Path, self).__setattr__(key, value)

    def join(self, *args):
        return Path(os.path.join(self, *args))

    def makedirs(self, exist_ok=True):
        '''Make directories of all children paths
        Be sure to define all folders first, makedirs(), and then define files in Path(),
        since defining files before makedirs() will lead to creating directories with names of files.
        It is possible to ignore paths with "." as all files do, but there are hidden directories that
        start with "." which makes things complicated. Thus, defining folders -> makedirs() -> define files
        is recommended.'''
        for directory in self.__dict__.values():
            if directory != '':
                os.makedirs(str(directory), exist_ok=exist_ok)
                if type(directory) == Path:
                    directory.makedirs(exist_ok=exist_ok)

    # DEPRECATED: Use shutil.rmtree(path) instead
    # def clear(self, ignore_errors=True):
    #     '''Delete all files and directories in current directory'''
    #     for directory in self.__dict__.values():
    #         shutil.rmtree(directory, ignore_errors=ignore_errors)

    def listdir(self, join=False, isdir=False, isfile=False):
        return _os.listdir(self, join=join, isdir=isdir, isfile=isfile)

class WindowsPath2(Path, pathlib.WindowsPath):
    pass
class PosixPath2(Path, pathlib.PosixPath):
    pass


class TDict(dict):
    '''
    Dictionary which can get items via attribute notation (class.attribute)

    Parameters
    ----------
    Identical to dict()

    Example
    -------

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, key):
        self.__delitem__(key)

class Printer():
    def __init__(self, filewrite_dir = None):
        self.content = ''
        self.filewrite_dir = filewrite_dir

    def add(self, text):
        self.content += text

    def print(self, *args, end='\n', flush=False):
        self.add(' '.join([str(arg) for arg in args]))
        print(self.content, end=end, flush=flush)
        if self.filewrite_dir != None:
            append(self.content + end, self.filewrite_dir)
        self.content=''

    def reset(self):
        self.content = ''

class Timer():
    '''
    Timer to measure elapsed time

    Parameters
    ----------
    print: bool (default: False)
        if True, then prints self.elapsed_time whenever stop() is called.

    return_f: bool (default: True)
        if True, then returns self.elapsed_time whenever stop() is called.

    auto_reset: bool (default: True)
        if True, then resets whenever start() is called.

    Methods
    -------
    start

    stop

    reset
        sets self.elapsed_time = 0
    '''
    def __init__(self, auto_reset = True):
        self.auto_reset = auto_reset
        self._elapsed_time = 0
        self.running = False

    def __repr__(self):
        options = ''
        if self.auto_reset: options += '(auto_reset)'

        if len(options)==0:
            return f'[Timer][running: {self.running}][elapsed_time: {self.elapsed_time()}]'
        else:
            return f'[Timer {options}][running: {self.running}][elapsed_time: {self.elapsed_time()}]'

    def __enter__(self):
        self.start()

    def __exit__(self):
        self.stop()

    def start(self):
        if self.auto_reset:
            # self.reset()
            self._elapsed_time = 0
        self.running = True
        self.start_time = time.time()

    def stop(self):
        if self.running:
            self.end_time = time.time()
            self._elapsed_time += self.end_time - self.start_time
            self.running = False
        return self._elapsed_time

    def elapsed_time(self):
        if self.running:
            return self._elapsed_time + time.time() - self.start_time
        else:
            return self._elapsed_time

    def reset(self):
        self._elapsed_time = 0

class Wrapper:
    def __repr__(self):
        name = '<wrapper>\n'
        args = 'args: '+' '.join(str(self.args))+'\n'
        kwargs = 'kwargs: '+str(self.kwargs)
        return name+args+kwargs

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = T.TDict(kwargs)
