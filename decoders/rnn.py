import torch
import torch.nn as nn
import torch.nn.functional as F

from __init__ import *

from global_config import device, MAX_LENGTH

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, padding_idx):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx = padding_idx)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output = torch.unsqueeze(output, 0)

        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, batch_size = 32):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class BahdanauDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, padding_idx, n_layers=1, drop_prob=0.1):
        super(BahdanauDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        self.embedding = nn.Embedding(self.output_size, self.hidden_size, padding_idx)

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(hidden_size, 1))
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.gru = nn.GRU(self.hidden_size*2, self.hidden_size, batch_first = True)
        self.classifier = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.squeeze()
        # Embed input words
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        # Calculating Alignment Scores
        hidden_state = self.fc_hidden(hidden[0])
        hidden_state = torch.stack([hidden_state for _ in range(encoder_outputs.shape[1])], dim = 1)
        x = torch.tanh(hidden_state + self.fc_encoder(encoder_outputs))

        alignment_scores = torch.einsum('ijk, kp -> ijp', x, self.weight)

        # Softmaxing alignment scores to get Attention weights
        attn_weights = F.softmax(alignment_scores, dim=1)

        # Multiplying the Attention weights with encoder outputs to get the context vector
        context_vector = torch.einsum('ijp, ijq -> iq', attn_weights, encoder_outputs)

        # Concatenating context vector with embedded input word
        output = torch.cat((embedded, context_vector), 1).unsqueeze(0)
        output = torch.transpose(output, 0, 1)

        # Passing the concatenated vector as input to the LSTM cell
        output, hidden = self.gru(output, hidden)
        # Passing the LSTM output through a Linear layer acting as a classifier

        output = F.log_softmax(self.classifier(output), dim=1)
        # output = torch.squeeze(output)

        return output, hidden, attn_weights

# class AdditiveAttention(nn.Module):
    # def __init__(self, ):
        # pass

    # def forward(self, decoder_hidden, encoder_hidden):


if __name__ == '__main__':
    import os
    import _pickle as pickle

    from global_config import DATA_DIR
    from textwrapper.dataset import Dataset
    from encoders.rnn import EncoderRNN

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
    batch_size = 32

    encoder = EncoderRNN(train_dataset.src_vocab.vocab_size, hidden_size, train_dataset.src_vocab.pad_idx).to(device)

    src = train_dataset[0][0]

    output, hidden = encoder(src, encoder.init_hidden())

    decoder = DecoderRNN(hidden_size, train_dataset.tgt_vocab.vocab_size, train_dataset.tgt_vocab.pad_idx)
    # attn_decoder = AttnDecoderRNN(hidden_size, test_dataset.tgt_vocab.vocab_size, train_dataset.tgt_vocab.pad_idx, dropout_p=0.1)
    attn_decoder = BahdanauDecoder(hidden_size, test_dataset.tgt_vocab.vocab_size, train_dataset.tgt_vocab.pad_idx)
    # decoder(torch.tensor([train_dataset.src_vocab.sos_idx for _ in range(batch_size)]), hidden)
    attn_decoder(torch.tensor([train_dataset.src_vocab.sos_idx for _ in range(batch_size)]), hidden, output)
