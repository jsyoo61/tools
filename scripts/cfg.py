import argparse
from tools.tools import tdict
from tools.tools import Path

lr=0.001
batch_size=256
n_epoch=50
n_layer=3

# Argument
try:
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--lr', type=float, default=lr, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=n_epoch, help='number of epochs')
    parser.add_argument('--n_layer', type=int, default=n_layer, help='number of layers')
    args = parser.parse_args()
except SystemExit as e:
    print(e)
    args = tdict()
    args.lr=lr
    args.batch_size=batch_size
    args.n_epoch=n_epoch
    args.n_layer=n_layer
print(args)

# Path
path = Path()
path.PROCESSED = '../processed/'
path.DATA_SPLIT = '../data_split/'
path.makedirs()
path()
