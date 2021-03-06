import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam
import codecs

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE
import json

def store_samples(iteration, samples):
    with codecs.open('data/samples-'str(iteration)+'.txt','w',encoding='utf-8') as fout:
        fout.write(samples)

def store_stats(stats):
    state_file = 'data/stats.json'
    mode = 'a' if os.path.exists(state_file) else 'w'
    with open(state_file, mode) as fout:
        fout.write(json.dumps(stats))    

def run(argument_list = None):

    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=120000, metavar='NI',
                        help='num iterations (default: 120000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--ce-result', default='', metavar='CE',
                        help='ce result path (default: '')')
    parser.add_argument('--kld-result', default='', metavar='KLD',
                        help='ce result path (default: '')')
    parser.add_argument('--author', default='', metavar='Author',
                        help='which author(folder) to use')

    if argument_list:
        args = parser.parse_args(argument_list)
    else:
        args = parser.parse_args()

    if not os.path.exists(args.author + '/word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    batch_loader = BatchLoader(args.author)
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE(parameters, args.author)
    if args.use_trained:
        rvae.load_state_dict(t.load('trained_RVAE'))
    if args.use_cuda:
        rvae = rvae.cuda()

    optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)

    train_step = rvae.trainer(optimizer, batch_loader)
    validate = rvae.validater(batch_loader)

    ce_result = []
    kld_result = []

    file = open('train_'+args.author+'.log', 'w')

    for iteration in range(args.num_iterations):

        cross_entropy, kld, coef = train_step(iteration, args.batch_size, args.use_cuda, args.dropout)

        if iteration % 5 == 0:
            file.write('\n')
            file.write('------------TRAIN-------------\n')
            file.write('----------ITERATION-----------\n')
            file.write(str(iteration))
            file.write('\n')
            file.write('--------CROSS-ENTROPY---------\n')
            file.write(str(cross_entropy.data.cpu().numpy()[0]))
            file.write('\n')
            file.write('-------------KLD--------------\n')
            file.write(str(kld.data.cpu().numpy()[0]))
            file.write('\n')
            file.write('-----------KLD-coef-----------\n')
            file.write(str(coef))
            file.write('\n')
            file.write('------------------------------\n')
            file.flush()

        if iteration % 10 == 0:
            cross_entropy, kld = validate(args.batch_size, args.use_cuda)

            cross_entropy = cross_entropy.data.cpu().numpy()[0]
            kld = kld.data.cpu().numpy()[0]

            file.write('\n')
            file.write('------------VALID-------------\n')
            file.write('--------CROSS-ENTROPY---------\n')
            file.write(str(cross_entropy))
            file.write('\n')
            file.write('-------------KLD--------------\n')
            file.write(str(kld))
            file.write('\n')
            file.write('------------------------------\n')
            file.flush()

            ce_result += [cross_entropy]
            kld_result += [kld]

        if iteration % 20 == 0:
            seed = np.random.normal(size=[1, parameters.latent_variable_size])

            sample = rvae.sample(batch_loader, 50, seed, args.use_cuda)

            file.write('\n')
            file.write('------------SAMPLE------------\n')
            file.write('------------------------------\n')
            file.write(str(sample))
            file.write('\n')
            file.write('------------------------------\n')
            file.flush()

    t.save(rvae.state_dict(), 'trained_RVAE')
    file.flush()
    file.close()

    np.save('ce_result_{}.npy'.format(args.ce_result), np.array(ce_result))
    np.save('kld_result_npy_{}'.format(args.kld_result), np.array(kld_result))

if __name__ == "__main__":
    run()
