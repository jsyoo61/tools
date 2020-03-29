import pickle
import time

def save_pickle(obj, path = None):
    if path == None:
        path = time.strftime('%y%m%d_%H%M%S.p')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def write(path, content, encoding = None):
    with open(path, 'w', encoding = encoding) as f:
        f.write(content)

def read(path, encoding = None):
    with open(path, 'r', encoding = encoding) as f:
        text = f.read()
    return text

def readlines(path, encoding = None):
    with open(path, 'r', encoding = encoding) as f:
        text = f.readlines()
    return text

def print_stars():
    print('*' * 50)

class Printer():
    def __init__(self):
        self.content = ''

    def add(self, text, end = '\n'):
        self.content += text + end

    def print(self,end='', flush=False):
        print(self.content, end=end, flush=flush)
        self.content=''

    def reset(self):
        self.content = ''
