import torch
import torch.nn as nn

from random import random
from torch import optim
from time import time

from __init__ import *
from encoders.rnn import EncoderRNN
from experiments.util import time_since
from decoders.rnn import DecoderRNN, BahdanauDecoder
from global_config import device, MAX_LENGTH

teacher_forcing_ratio = 0.5

def train_datum(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, sos_tok, eos_tok, batch_size, max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_input = torch.tensor([sos_tok for _ in range(batch_size)], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random() < teacher_forcing_ratio else False

    for di in range(target_tensor.shape[1]):
        print(di)
        tgt = target_tensor[:, di].unsqueeze(1)
        decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
        # print(decoder_output.shape)
        decoder_output = torch.transpose(decoder_output, 1, 2)

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            print(decoder_input.shape, decoder_output.shape, tgt.shape)
            loss += criterion(decoder_output, tgt)
            decoder_input = target_tensor[di]  # Teacher forcing
            print(decoder_input.shape, 2)
        else:
            # Without teacher forcing: use its own predictions as the next input

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            print(decoder_input.shape, decoder_output.shape, tgt.shape)

            loss += criterion(decoder_output, tgt)
        print(loss.item())

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    # print(loss.item())
    return loss.item() / target_length

def train(train_data, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    train_data_size = len(train_data.src_data)
    sos_tok = train_data.tgt_vocab.sos_idx
    eos_tok = train_data.tgt_vocab.eos_idx
    batch_size = train_data.batch_size

    for iter in range(1, n_iters + 1):
        training_pair = train_data[iter % train_data_size]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train_datum(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, sos_tok, eos_tok, batch_size)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

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
    attn_decoder = BahdanauDecoder(hidden_size, test_dataset.tgt_vocab.vocab_size, train_dataset.tgt_vocab.pad_idx)

    train(train_dataset, encoder, attn_decoder, 75000, print_every=5000)

