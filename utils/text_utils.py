
from collections import Counter
from string import punctuation

import pandas as pd
import numpy as np


def read_in_csv_data(csv):

    data = pd.read_csv(csv)
    print('[i] Reading in dataaset....')
    text = list(data['review_text'])
    labels = list(data['sentiment'])

    return text, labels


def tokenizer(text):

    text = text.lower()
    all_text = ''.join([char for char in text if char not in punctuation])

    text_split = all_text.split('\n')
    all_text = ' '.join(text_split)

    tokens = all_text.split()
    print(f'[+] Number of Tokens: {len(tokens)}')

    return tokens, text_split


