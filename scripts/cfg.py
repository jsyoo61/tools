import argparse
from tools.tools import tdict
from tools.tools import Path

# Argument
try:
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers')
    args = parser.parse_args()
except:
    args = tdict()
    args.lr=0.0001
    args.batch_size=256
    args.n_epochs=50
    args.n_layers=3
print(args)

# Path
path = Path()
path.PROCESSED = '../processed/'
path.DATA_SPLIT = '../data_split/'
path.makedirs()
path()
