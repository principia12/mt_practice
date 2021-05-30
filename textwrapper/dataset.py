import torch
from collections import defaultdict

from random import random

from __init__ import *

from textwrapper.config import UNK, EOS, SOS, PAD
from textwrapper.vocab import Vocab

class Dataset:
    def __init__(self, src_data,
                    tgt_data,
                    src_vocab,
                    tgt_vocab,
                    batch_size = 32, ):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.batch_size = batch_size

        self.data = Dataset._preprocess(src_data, tgt_data,
                                    src_vocab, tgt_vocab, batch_size)

    def __iter__(self):
        yield from self.data

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def _preprocess(src_data, tgt_data, src_vocab, tgt_vocab, batch_size):
        res = []

        for batch_num in range(len(src_data) // batch_size):
            src = src_data[batch_num * batch_size:(batch_num + 1)*batch_size]
            tgt = tgt_data[batch_num * batch_size:(batch_num + 1)*batch_size]
            src = src_vocab.preprocess(src)
            tgt = tgt_vocab.preprocess(tgt)

            # src = torch.transpose(src_vocab.preprocess(src), 0, 1)
            # tgt = torch.transpose(tgt_vocab.preprocess(tgt), 0, 1)

            res.append((src, tgt))

        return res

    @staticmethod
    def build_from_file(file_path, batch_size = 32,
                        train_valid_test_ratio = (0.8, 0.1, 0.1)):
        train_src_data = []
        valid_src_data = []
        test_src_data = []

        train_tgt_data = []
        valid_tgt_data = []
        test_tgt_data = []

        src_data, tgt_data = [], []
        l = train_valid_test_ratio

        with open(file_path, 'r', encoding = 'utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                src, tgt = line.split('\t')

                opt = random()

                if opt < l[0]:
                    train_src_data.append(src)
                    train_tgt_data.append(tgt)
                elif l[0] < opt < sum(l[:1]):
                    valid_src_data.append(src)
                    valid_tgt_data.append(tgt)
                else:
                    test_src_data.append(src)
                    test_tgt_data.append(tgt)

                src_data.append(src)
                tgt_data.append(tgt)

        src_vocab = Vocab()
        tgt_vocab = Vocab()

        src_vocab.build(src_data)
        tgt_vocab.build(tgt_data)

        train_dataset = Dataset(train_src_data, train_tgt_data, src_vocab, tgt_vocab, batch_size = batch_size)
        valid_dataset = Dataset(valid_src_data, valid_tgt_data, src_vocab, tgt_vocab, batch_size = batch_size)
        test_dataset = Dataset(test_src_data, test_tgt_data, src_vocab, tgt_vocab, batch_size = batch_size)

        return train_dataset, valid_dataset, test_dataset


if __name__ == '__main__':
    import os
    from global_config import DATA_DIR
    train_dataset, valid_dataset, test_dataset = Dataset.build_from_file(os.path.join(DATA_DIR, 'eng-fra.txt'))

    for d in train_dataset:
        print(d[0].shape)