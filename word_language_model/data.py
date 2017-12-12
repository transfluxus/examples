import os
import torch
from itertools import chain 
from gensim.corpora.dictionary import Dictionary

class Corpus(object):
    def __init__(self, path, dict_path):
        self.dictionary = Dictionary()
        if dict_path and os.path.exists(dict_path):
                print('loading dictionary')
                self.dictionary.load(dict_path)
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        if dict_path and not os.path.exists(dict_path):
            self.dictionary.save(dict_path)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        all_words = list(chain.from_iterable(
            [sent.split() + ['<eos>'] for sent in open(path).read().split('\n')]))
        self.dictionary.add_documents([all_words])
        return torch.LongTensor(self.dictionary.doc2idx(all_words))

