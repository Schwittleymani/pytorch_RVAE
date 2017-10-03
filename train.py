import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam
import codecs

import json

from pytorch_RVAE.utils.batch_loader import BatchLoader
from pytorch_RVAE.utils.parameters import Parameters
from pytorch_RVAE.model.rvae import RVAE

def store_samples(iteration, samples):
    with codecs.open(unicode('pytorch_RVAE/data/samples-' + str(iteration)+'.txt', 'utf-8'),'w',encoding='utf-8') as fout:
        fout.write(unicode(samples, 'utf-8'))

def store_stats(stats):
    state_file = 'pytorch_RVAE/data/stats.json'
    mode = 'a' if os.path.exists(state_file) else 'w'
    with open(state_file, mode) as fout:
        print(stats)
        print(type(stats))
        json.dump(stats, fout)
        #fout.write(json.dumps(stats))

def run(argument_list = None):

    if not os.path.exists('pytorch_RVAE/data/word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    batch_loader = BatchLoader('pytorch_RVAE/')
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE(parameters)
    rvae = rvae.cuda()

    optimizer = Adam(rvae.learnable_parameters(), 0.00005)

    train_step = rvae.trainer(optimizer, batch_loader)
    validate = rvae.validater(batch_loader)

    ce_result = []
    kld_result = []

    iters = 500  # 1000000
    for iteration in range(iters):

        cross_entropy, kld, coef = train_step(iteration, 32, True, 0.3)

        if iteration % 5 == 0:
            # print('\n')
            # print('------------TRAIN-------------')
            # print('----------ITERATION-----------')
            # print(iteration)
            # print('--------CROSS-ENTROPY---------')
            cross_entropy_data_cpu = cross_entropy.data.cpu().numpy()[0]
            # print(cross_entropy_data_cpu)
            # print('-------------KLD--------------')
            # print(kld.data.cpu().numpy()[0])
            # print('-----------KLD-coef-----------')
            # print(coef)
            # print('------------------------------')
            stats = {
                "it": float(iteration),
                "cedc": float(cross_entropy_data_cpu),
                "kld": float(kld.data.cpu().numpy()[0]),
                "coef": float(kld.data.cpu().numpy()[0])
            }
            store_stats(stats)

        if iteration % 10 == 0:
            cross_entropy, kld = validate(32, True)

            cross_entropy = cross_entropy.data.cpu().numpy()[0]
            kld = kld.data.cpu().numpy()[0]

            # print('\n')
            # print('------------VALID-------------')
            # print('--------CROSS-ENTROPY---------')
            # print(cross_entropy)
            # print('-------------KLD--------------')
            # print(kld)
            # print('------------------------------')
            stats = {
                "it": float(iteration),
                "ce": float(cross_entropy),
                "kld": float(kld)
            }
            store_stats(stats)

            ce_result += [cross_entropy]
            kld_result += [kld]

        if iteration % 20 == 0:
            seed = np.random.normal(size=[1, parameters.latent_variable_size])

            sample = rvae.sample(batch_loader, 50, seed, True)

            # print('\n')
            # print('------------SAMPLE------------')
            # print('------------------------------')
            # print(sample)
            store_samples(iteration, sample)
            # print('------------------------------')
        if iteration % 100 == 0:
            print(iteration)


    t.save(rvae.state_dict(), 'trained_RVAE')

    np.save('ce_result_{}.npy'.format(''), np.array(ce_result))
    np.save('kld_result_npy_{}'.format(''), np.array(kld_result))

if __name__ == "__main__":
    run()
