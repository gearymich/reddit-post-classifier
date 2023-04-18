# https://www.kaggle.com/code/swarnabha/pytorch-text-classification-torchtext-lstm/notebook

import pandas as pd
import numpy as np
import torch
import torchtext

SEED = 42

from sklearn.model_selection import train_test_split

super_categories = ['Sports', 'Politics', 'Entertainment', 'News', 'Science']

if __name__ == '__main__':
    # tr, te = load_data(path='reddit_with_super_categories.csv')
    # print(te.head())

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    str_1 = 'hello world'
    # tokenize the text
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    print(tokenizer(str_1))

    # create a vocabulary
    vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, [str_1]))