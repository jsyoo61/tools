import os as os
import pickle
import time
import subprocess
from pathlib import Path as P
import yaml

__all__ = \
['AverageMeter',
 'Filename',
 'Printer',
 'Timer',
 'Path',
 'append',
 'cmd',
 'is_iterable',
 'load_pickle',
 'now',
 'prettify_dict',
 'print_dict',
 'read',
 'readline',
 'readlines',
 'save_pickle',
 'str2bool',
 'tdict',
 'write']

def save_pickle(obj: str, path: str = None):
    '''Save object as Pickle file to designated path.
    If path is not given, default to "YearMonthDay_HourMinuteSecond.p" '''
    if path == None:
        path = time.strftime('%y%m%d_%H%M%S.p')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    '''Load Pickle file from designated path'''
    with open(path, 'rb') as f:
        return pickle.load(f)

def write(content, path, encoding = None):
    with open(path, 'w', encoding = encoding) as f:
        f.write(content)

def append(content, path, encoding = None):
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

def cmd(command: str):
    '''Run shell command'''
    pipe = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
    return pipe.stdout

def prettify_dict(dictionary, indent=0):
    return '\n'.join([' '*indent + str(k) +': '+str(v) if type(v)!=dict else str(k)+':\n'+prettify_dict(v, indent=indent+2) for k, v in dictionary.items()])

# Used in Argparse
def str2bool(x):
    true_list = ['t', 'true', 'y', 'yes', '1']
    false_list = ['f', 'false', 'n', 'no', '0']
    if x.lower() in true_list:
        return True
    elif x.lower() in false_list:
        return False
    else:
        raise Exception('input has to be in one of two forms:\nTrue: %s\nFalse: %s'%(true_list, false_list))

def is_iterable(x):
    return hasattr(x, '__iter__')

def now(format: str ='-'):
    if format=='-':
        return time.strftime('%Y-%m-%d_%H-%M-%S')
    elif format=='_':
        return time.strftime('%y%m%d_%H%M%S')
    else:
        raise Exception("format has to be one of ['-', '_']")


def print_dict(d, **kwargs):
    '''prettify long dictionary
    Example
    -------
    >>> d = {'a':1, 'b':2, 'c':3}
    >>> print_dict(d)

    '''

    return yaml.dumps(d, **kwargs)

class Filename():
    '''(Will be deprecated since Path() can be used)
    Class to handle Filename with suffix.
    To call filename with suffix, call the instance.

    Example
    -------
    >>> file = Filename('model', '.pt')
    >>> file
    (Filename, name: "model", suffix: ".pt")
    >>> str(file)
    "model"
    >>> file()
    "model.pt"
    >>> (file+'1')
    "model1"
    >>> (file+'1')()
    "model1.pt"
    '''
    def __init__(self,obj=None, suffix=''):
        self.name = str(obj)
        self.suffix = suffix

    def __call__(self):
        '''Add suffix and return str'''
        return self.name+self.suffix

    def __radd__(self, other):
        return Filename(other+self.name,suffix=self.suffix)

    def __add__(self, other):
        return Filename(self.name+other,suffix=self.suffix)

    def __mul__(self, other):
        return Filename(self.name*other,suffix=self.suffix)

    def __repr__(self):
        return '(Filename, name: "%s", suffix: "%s")'%(self.name, self.suffix)

    def __str__(self):
        return self.name

class Path(str):
    '''
    Joins paths by . syntax
    Use pathlib.Path internally

    Parameters
    ----------
    path: str (default: '.')
        Notes the default path. Leave for default blank value which means the current working directory.

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
    def __init__(self, path='.'):
        self.path=path

    def __repr__(self):
        return self.path


    def __call__(self):
        '''Print out current path, and children'''
        for name, directory in self.__dict__.items():
            if name != 'path':
                print(name+': '+str(directory))
                if type(directory) == Path:
                    directory()
        # print('\n'.join([key+': '+str(value) for key, value in self.__dict__.items()]))

    def __str__(self):
        return self.path

    def __setattr__(self, key, value):
        # super(Path, self).__setattr__(key, self / value) # self.joinpath(value)
        if hasattr(self, 'path'):
            super(Path, self).__setattr__(key, Path(os.path.join(self.path, value)))
        else:
            super(Path, self).__setattr__(key, value)

    def join(self, *args):
        return os.path.join(self.path, *args)

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

    def clear(self, ignore_errors=True):
        '''Delete all files and directories in current directory'''
        for directory in self.__dict__.values():
            shutil.rmtree(directory, ignore_errors=ignore_errors)


class tdict(dict):
    '''
    Dictionary which can get items via attribute notation (class.attribute)

    Parameters
    ----------
    Identical to dict()

    Example
    -------

    '''
    def __init__(self, *args, **kwargs):
        super(tdict, self).__init__(*args, **kwargs)

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
    '''Timer to measure elapsed time

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
    def __init__(self, print = False, return_f = True, auto_reset = True):
        self.print = print
        self.return_f = return_f
        self.auto_reset = auto_reset
        self.elapsed_time = 0
        self.running = False

    def __enter__(self):
        self.start()

    def __exit__(self):
        t = self.stop()
        print(t)

    def start(self):
        if self.auto_reset:
            # self.reset()
            self.elapsed_time = 0
        self.running = True
        self.start_time = time.time()

    def stop(self):
        if self.running:
            self.end_time = time.time()
            self.elapsed_time += self.end_time - self.start_time
            self.running = False
        if self.print:
            print(self.elapsed_time)
        if self.return_f:
            return self.elapsed_time

    def reset(self):
        self.elapsed_time = 0

class AverageMeter(object):
    """Computes and stores the average and current value
    Variables
    ---------
    self.val
    self.avg
    self.sum
    self.count
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def step(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
