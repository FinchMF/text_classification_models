
from collections import Counter
from string import punctuation

import pandas as pd
import numpy as np

import torch

def read_in_csv_data(csv):

    data = pd.read_csv(csv)
    print('[i] Reading in dataaset....')
    text = list(data['review_text'])
    labels = list(data['sentiment'])

    return text, labels

def tokenizer(text):

    text = ''.join(text)
    text = text.lower()
    all_text = ''.join([char for char in text if char not in punctuation])

    text_split = all_text.split('\n')
    all_text = ' '.join(text_split)

    tokens = all_text.split()
    print(f'[+] Number of Tokens: {len(tokens)}')

    return tokens, text_split

def encode_tokens(tokens, text_split):

    counts = Counter(tokens)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {token: ii for ii, token in enumerate(vocab, 1)}

    text_ints = []
    for text in text_split:
        text_ints.append([vocab_to_int[token] for token in text.split()])

    return vocab_to_int, text_ints

def encode_labels(labels):

    encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels])
    
    return encoded_labels

def pad_features(text_ints, seq_length):

    features = np.zeros((len(text_ints), seq_length), dtype=int)

    for i, row in enumerate(text_ints):
        features[i, -len(row): ] = np.array(row)[:seq_length]

    return features




def tokenize_input(test, vocab_to_int):

    test = test.lower()
    text = ''.join([char for char in test if char not in punctuation])
    words = text.split()

    tokens = []
    tokens.append([vocab_to_int[word] for word in words])

    return tokens


def predict(net, test, vocab_to_int, seq_length, pad_features):

    net.eval()
    tokens = tokenize_input(test, vocab_to_int)

    seq_length=seq_length
    features = pad_features(tokens, seq_length)
    feature_tensor = torch.from_numpy(features)
    batch_size = feature_tensor.size(0)

    h = net.init_hidden(batch_size)

    train_on_gpu=torch.cuda.is_available()

    if train_on_gpu:

        feature_tensor = feature_tensor.cuda()
    output, h = net(feature_tensor, h)
    pred = torch.round(output.squeeze())
    print('Prediction value, pre-rounding: {:.6}'.format(output.item()))

    if pred.item() == 1:
        print('positve')
    else:
        print('negative')



