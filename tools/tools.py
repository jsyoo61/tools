import pickle
import time
import subprocess

def save_pickle(obj, path = None):
    if path == None:
        path = time.strftime('%y%m%d_%H%M%S.p')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
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

def cmd(command):
    pipe = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
    return pipe.stdout

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
# Use '*' * 30
# def stars():
#     return '*' * 30

# Use print('*' * 30)
# def print_stars():
#     print('*' * 30)

# Use dict.update()
# def update_dict(dict, new_dict):
#     for key, value in new_dict.items:
#         dict[key] = value
#     return dict

class Printer():
    def __init__(self, filewrite_dir = None):
        self.content = ''
        self.filewrite_dir = filewrite_dir

    def add(self, text):
        self.content += text

    def print(self, *args, end='\n', flush=False):
        self.add(' '.join([str(arg) for arg in args]))
        print(self.content, end=end, flush=flush)
        if self.filewrite_dir is not None:
            append(self.content + end, self.filewrite_dir)
        self.content=''

    def reset(self):
        self.content = ''

class Timer():
    '''Timer to measure elapsed time

    Parameters
    ----------
    print: bool (default: False)
        if True, then prints self.elapsed_time whenever stop() is called

    return: bool (default: True)
        if True, then returns self.elapsed_time whenever stop() is called

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
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
