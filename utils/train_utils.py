from collections import Counter
import numpy as np 

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils.utils_params import configure_utility_params, configure_LSTM, configure_training_params 
from utils.text_utils import encode_tokens, encode_labels, tokenizer, pad_features, read_in_csv_data, predict

class Dataset:

    def __init__(self, root):

        self.text, self.labels = read_in_csv_data(root)
        self.tokens, self.text_split = tokenizer(self.text)

        self.vocab_to_int, self.text_ints = encode_tokens(self.tokens, self.text_split)
        self.encode_labels = encode_labels(self.labels)

        self.text_lens = Counter([len(x) for x in self.text_ints])
        

def extract_features(root):  

    D = Dataset(root)

    non_zero_idx = [ii for ii, text in enumerate(D.text_ints) if len(text) != 0]
    text_ints = [D.text_ints[ii] for ii in non_zero_idx]
    encoded_labels = np.array([D.encode_labels[ii] for ii in non_zero_idx])

    print('\n')
    print(f'[i] Number of Text after removing outliers: {len(text_ints)}')
    print('\n')

    utility_params = configure_utility_params()

    features = pad_features(text_ints, utility_params['seq_length'])

    assert len(features) == len(text_ints)
    assert len(features[0]) == utility_params['seq_length']

    return features, encoded_labels

def split_feature_data(features, encoded_labels):

    print(f'Number of Features: {len(features)}')
    print(f'Number of Encoded Labels: {len(encoded_labels)}')

    utility_params = configure_utility_params()

    split_idx = int(len(features)*utility_params['split_frac'])
    print(f'Split Index: {split_idx}')

    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]
    print(f'[+] Validation / Test Y: {len(remaining_y)}')

    test_idx = int(len(remaining_x)*0.5)
    print(f'[+] Test Index: {test_idx}')

    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

    print('\t\t\t[i] Feature Shapes:')
    print(f'[i] Train set: \t\t{train_x.shape}',
        f'\n[i] Validation set: \t{val_x.shape}',
        f'\n[i] Test set: \t\t{test_x.shape}')
    print('\n')
    print('\t\t\t[i] Label Shapes:')
    print(f'[i] Train set: \t\t{train_y.shape}',
        f'\n[i] Validation set: \t{val_y.shape}',
        f'\n[i] Test set: \t\t{test_y.shape}')
    print('\n')

    return train_x, train_y, val_x, val_y, test_x, test_y

def transform_split_data(train_x, train_y, val_x, val_y, test_x, test_y):

    utility_params = configure_utility_params()

    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=utility_params['batch_size'], drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=utility_params['batch_size'], drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=utility_params['batch_size'], drop_last=True)

    return train_data, valid_data, test_data, train_loader, valid_loader, test_loader

def generate_sample(train_loader):

    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()

    print(f'[i] Sample Input size: {sample_x.size()}')
    print(f'[i] Sample Input: \n {sample_x}')
    print('\n')
    print(f'[i] Sample Label size: {sample_y.size()}')
    print(f'[i] Sample Label: \n {sample_y}')
    print('\n')

    return sample_x, sample_y

def train_loop(net, train_loader, valid_loader, name):

    print(f'[+] Model Configuration: {net}')

    training_params = configure_training_params(net)
    utility_params = configure_utility_params()
    train_on_gpu = torch.cuda.is_available()

    if train_on_gpu:
        net.cuda()

    counter = 0
    net.train()

    for e in range(training_params['epochs']):

        h = net.init_hidden(utility_params['batch_size'])

        for inputs, labels in train_loader:
            counter += 1

            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            h = tuple([each.data for each in h])

            net.zero_grad()

            output, h = net(inputs, h)
            loss = training_params['criterion'](output.squeeze(), labels.float())
            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), training_params['clip'])
            training_params['optimizer'].step()

            if counter % training_params['print_every'] == 0:

                val_h = net.init_hidden(utility_params['batch_size'])
                val_losses = []
                net.eval()

                for inputs, labels in valid_loader:

                    val_h = tuple([each.data for each in val_h])

                    if train_on_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = net(inputs, val_h)

                    val_loss = training_params['criterion'](output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())

                    net.train()
                    print(f'Epoch: {e+1}/{training_params["epochs"]}',
                            f'Step: {counter}',
                            'Loss: {:.6f}...'.format(loss.item()),
                            'Val Loss: {:.6}'.format(np.mean(val_losses)))

    torch.save(net.state_dict, f'./models/gen_models/{name}_paramsDict.pth')
    print('LSTM Classifier\'s parameters saved....')
    torch.save(net, f'./models/gen_models/{name}_lstm_classifer.pth')
    print('LSTM Classifier saved....') 

    return None

    

def training_pipeine(net, root, name):

    features, encoded_labels = extract_features(root)
    train_x, train_y, val_x, val_y, test_x, test_y = split_feature_data(features, encoded_labels)
    train_data, valid_data, test_data, train_loader, valid_loader, test_loader = transform_split_data(train_x, train_y, val_x, val_y, test_x, test_y)
    sample_x, sample_y = generate_sample(train_loader)

    print(f'Sample Input size: {sample_x.size()}')
    print(f'Sample Input: \n {sample_x}')
    print('\n')
    print(f'Sample Label size: {sample_y.size()}')
    print(f'Sample Label: \n {sample_y}')
    print('\n')

    train_loop(net, train_loader, valid_loader, name)

    return print('[i] Finish Training')











                        








