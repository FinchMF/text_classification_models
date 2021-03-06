
import utils.train_utils as tu 
import utils.utils_params as up 

from models.LSTM import LSTM_Classifier


def run_training_loop():

    root, name = up.configure_path_params()

    Data = tu.Dataset(root)
    print('[i] Data Object Set...')
    lstm_params = up.configure_LSTM(Data.vocab_to_int)
    print('[i] Parameters Set...')
    net = LSTM_Classifier(lstm_params['vocab_size'],
                            lstm_params['output_size'],
                            lstm_params['embedding_dim'],
                            lstm_params['hidden_dim'],
                            lstm_params['n_layers'])
    print('[+] Model Set...')
    print('[+] Begin Training...')
    tu.training_pipeine(net, root, name)

    return print('Saved')


if __name__ == '__main__':

    run_training_loop()


    


    


    
    

