import torch
from collections import defaultdict

from __init__ import *

from global_config import MAX_LENGTH
from textwrapper.config import UNK, EOS, SOS, PAD
from textwrapper.util import default_tokenizer, default_detokenizer

class Vocab:
    def __init__(self, min_freq = 0, tokenizer = default_tokenizer, detokenizer = default_detokenizer ):
        self.min_freq = min_freq
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer

        self.bow = defaultdict(int)
        self.itos = {0: UNK, 1: EOS, 2: SOS, 3: PAD}
        self.stoi = {UNK: 0, EOS: 1, SOS: 2, PAD: 3}

        assert len(self.itos) == len(self.stoi)
        self.vocab_size = len(self.itos)

        self.unk = UNK
        self.eos = EOS
        self.sos = SOS
        self.pad = PAD

        self.unk_idx = self.stoi[UNK]
        self.eos_idx = self.stoi[EOS]
        self.sos_idx = self.stoi[SOS]
        self.pad_idx = self.stoi[PAD]

    def build(self, sents):

        for sent in sents:
            tokens = self.tokenizer(sent)

            for tok in tokens:
                self.bow[tok] += 1

        for k, v in self.bow.items():
            if v > self.min_freq:
                if k not in self.stoi:
                    self.stoi[k] = self.vocab_size
                    self.itos[self.vocab_size] = k
                    self.vocab_size += 1

    def build_from_file(self, file):
        with open(file, 'r') as f:

            for line in f.readlines():
                k = line.strip()
                self.stoi[k] = self.vocab_size
                self.itos[self.vocab_size] = k
                self.vocab_size += 1

    def preprocess(self, sents, max_length = MAX_LENGTH):
        """Sents to torch tensor for input.
        """
        res = []

        for sent in sents:
            res.append([self.sos_idx])
            tokens = self.tokenizer(sent)

            for tok in tokens[:max_length]:
                if tok in self.stoi:
                    res[-1].append(self.stoi[tok])
                else:
                    res[-1].append(self.unk_idx)
            else:
                for _ in range(max_length - len(tokens)):
                    res[-1].append(self.pad_idx)
                res[-1].append(self.eos_idx)

        return torch.tensor(res)

    def sentize(self, inference_result):
        sents = []

        for tokens in inference_result:
            sent = []
            for tok in sent:
                sents.append(self.itos[tok])

if __name__ == '__main__':
    pass