import os
import torch
from itertools import chain 
from gensim.corpora.dictionary import Dictionary

class Corpus(object):
    def __init__(self, path, dict_path):
        self.dictionary = Dictionary()
        add_to_dict = True
        if dict_path and os.path.exists(dict_path):
            print('loading dictionary')
            self.dictionary = self.dictionary.load(dict_path)
            add_to_dict = False
        self.train = self.tokenize(os.path.join(path, 'train.txt'), add_to_dict)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), add_to_dict)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), add_to_dict)
        if dict_path and not os.path.exists(dict_path):
            self.dictionary.save(dict_path)

    def tokenize(self, path, add_to_dict):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        all_words = list(chain.from_iterable(
            [sent.split() + ['<eos>'] for sent in open(path).read().split('\n')]))
        if add_to_dict:
            self.dictionary.add_documents([all_words])
        return torch.LongTensor(self.dictionary.doc2idx(all_words))

