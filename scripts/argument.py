import argparse

parser = argparse.ArgumentParser(description='description')
parser.add_argument('--argument', type=eval, default=1, help='help')
args = parser.parse_args()
