import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, word_embedding = None):
        super(RNNModel, self).__init__()
        self.wem = word_embedding
        self.drop = nn.Dropout(dropout)
        if not word_embedding:          
            self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        if not word_embedding:
            self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights and not self.wem:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        if not self.wem:
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.bias.data.fill_(0)
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        # print("input",input.data.shape)
        if not self.wem:
            enc = self.encoder(input)
            print('enc',enc.data.shape)
        else:
            # here transform input: 35*20:word_idx > 35*20*200:word_vecs
            # make it 700:w_idx > create new 700*200, loop to fill 
            # PROBLEM WITH NEW:  LOSING VARIABLE
            # enc = torch.FloatTensor(self.wem.wv.syn0)
            enc = input
        emb = self.drop(enc)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        # print('output',output.data.shape)
        if not self.wem:
            decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
            print('decoded',decoded.data.shape)
            ret = decoded.view(output.size(0), output.size(1), decoded.size(1))
            print('ret',ret.data.shape)
            return ret, hidden
        else:
            return output,hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
