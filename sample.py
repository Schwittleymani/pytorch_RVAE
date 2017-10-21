import argparse
import os

import numpy as np
import torch as t

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE

if __name__ == '__main__':

    assert os.path.exists('trained_RVAE'), \
        'trained model not found'

    parser = argparse.ArgumentParser(description='Sampler')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--num-sample', type=int, default=100000, metavar='NS',
                        help='num samplings (default: 10)')
    parser.add_argument('--author', default='', metavar='Author',
                        help='which author(folder) to use')
    args = parser.parse_args()

    batch_loader = BatchLoader(args.author)
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE(parameters, args.author)
    rvae.load_state_dict(t.load('trained_RVAE'))
    if args.use_cuda:
        rvae = rvae.cuda()
    file = open(args.author + '-100000samples.txt', 'w')
    for iteration in range(args.num_sample):
        seed = np.random.normal(size=[1, parameters.latent_variable_size])
        result = rvae.sample(batch_loader, 50, seed, args.use_cuda)
        print(result)
        file.write(result + '\n')
        file.flush()
    print()
    file.close()
