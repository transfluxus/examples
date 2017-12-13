import os
import torch
from itertools import chain 
from gensim.corpora.dictionary import Dictionary
import numpy as np

class Corpus(object):
    def __init__(self, path, dict_path, wem_model):
        self.wem_model = wem_model
        if not wem_model:
            self.dictionary = Dictionary()
            add_to_dict = True
            if dict_path and os.path.exists(dict_path):
                print('loading dictionary')
                self.dictionary = self.dictionary.load(dict_path)
                add_to_dict = False
        else:
            add_to_dict = False
        self.train = self.tokenize(os.path.join(path, 'train.txt'), add_to_dict)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), add_to_dict)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), add_to_dict)

        if not self.wem_model:
            print('dictionary size',len(self.dictionary))
            if dict_path and not os.path.exists(dict_path):
                self.dictionary.save(dict_path)

    def tokenize(self, path, add_to_dict):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        all_words = list(chain.from_iterable(
            [sent.split() + ['<eos>'] for sent in open(path).read().split('\n')]))
        if add_to_dict:
            self.dictionary.add_documents([all_words])
        if self.wem_model:
            return torch.from_numpy(np.array([self.wem_model.wv[word] for word in all_words]))
        else:
            return torch.LongTensor(self.dictionary.doc2idx(all_words))


    def dict_size(self):
        if not self.wem_model:
            return len(self.dictionary)
        else:
            return len(self.wem_model.wv.syn0)
