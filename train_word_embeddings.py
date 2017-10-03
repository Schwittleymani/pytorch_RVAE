import argparse

import numpy as np
import torch as t
from torch.autograd import Variable
from torch.optim import SGD

from pytorch_RVAE.utils.batch_loader import BatchLoader
from pytorch_RVAE.utils.parameters import Parameters
from pytorch_RVAE.selfModules.neg import NEG_loss

def run():
    batch_loader = BatchLoader('pytorch_RVAE/')
    params = Parameters(batch_loader.max_word_len,
                        batch_loader.max_seq_len,
                        batch_loader.words_vocab_size,
                        batch_loader.chars_vocab_size)

    neg_loss = NEG_loss(params.word_vocab_size, params.word_embed_size)
    neg_loss = neg_loss.cuda()

    # NEG_loss is defined over two embedding matrixes with shape of [params.word_vocab_size, params.word_embed_size]
    optimizer = SGD(neg_loss.parameters(), 0.1)

    iters = 500 #1000000
    for iteration in range(iters):

        input_idx, target_idx = batch_loader.next_embedding_seq(5)

        input = Variable(t.from_numpy(input_idx).long())
        target = Variable(t.from_numpy(target_idx).long())
        input, target = input.cuda(), target.cuda()

        out = neg_loss(input, target, 5).mean()

        optimizer.zero_grad()
        out.backward()
        optimizer.step()

        if iteration % 500 == 0:
            out = out.cpu().data.numpy()[0]
            print('iteration = {}, loss = {}'.format(iteration, out))

    word_embeddings = neg_loss.input_embeddings()
    np.save('pytorch_RVAE/data/word_embeddings.npy', word_embeddings)

if __name__ == '__main__':
    run()

