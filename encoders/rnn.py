import torch
import torch.nn as nn

from __init__ import *
from global_config import device

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, padding_idx):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx = padding_idx)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, batch_size = 32):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

if __name__ == '__main__':
    import os
    import _pickle as pickle

    from global_config import DATA_DIR
    from textwrapper.dataset import Dataset

    if 'train.pickle' in os.listdir():
        train_dataset = pickle.load(open('train.pickle', 'rb'))
        valid_dataset = pickle.load(open('valid.pickle', 'rb'))
        test_dataset = pickle.load(open('test.pickle', 'rb'))
    else:
        train_dataset, valid_dataset, test_dataset = Dataset.build_from_file(os.path.join(DATA_DIR, 'eng-fra.txt'))
        pickle.dump(train_dataset, open('train.pickle', 'wb+'))
        pickle.dump(valid_dataset, open('valid.pickle', 'wb+'))
        pickle.dump(test_dataset, open('test.pickle', 'wb+'))

    hidden_size = 256
    encoder = EncoderRNN(train_dataset.src_vocab.vocab_size, hidden_size, train_dataset.src_vocab.pad_idx).to(device)

    for src, tgt in train_dataset:
        encoder(src, encoder.init_hidden())
