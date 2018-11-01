import argparse
import torchfile
import numpy as np
import pickle
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--t7_path', type=str, default='')

args = parser.parse_args()

print('load t7 from ', args.t7_path)
save_path = args.t7_path[:-3]
save_path += '.pickle'

data = torchfile.load(args.t7_path)
res = []
assert(len(data[b'fea_txt']) == len(data[b'raw_txt']))
for e, t in zip(data[b'fea_txt'], data[b'raw_txt']):
    res.append({'embedding':e, 'caption': t})

print (len(res), 'files are fouded')
print('save pickle to  ', save_path)

pickle.dump(res, open(save_path,'wb'))
