import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, nhid, nlayers, dropout=0.5, word_embedding = None):
        super(RNNModel, self).__init__()
        self.wem = word_embedding
        # self.drop = nn.Dropout(dropout)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(word_embedding.vector_size, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(word_embedding.vector_size, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.LOut = nn.Linear(nhid, word_embedding.vector_size)
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers


    def forward(self, input, hidden):
        # self.rnn.flatten_parameters()
        output, hidden = self.rnn(input, hidden)
        # output = self.drop(output)
        # print('output',output.data.shape)
        output = self.LOut(output)
        return output,hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
