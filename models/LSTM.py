

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F 

class LSTM_Classifier(nn.Module):

    def __init__(self, batch_size, output_size, hidden_size, vocab_size, emedding_dim, weights):
        super(LSTM_Classifier, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = emedding_dim
        self.weights = weights

        self.word_embeddings = nn.Embedding(vocab_size, emedding_dim)
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size)
        self.label = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_sentence, batch_size=None):

        inpt = self.word_embeddings(input_sentence)
        inpt = inpt.permute(1, 0, 2)

        if batch_size is None:

            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())

        else:

            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

        output_size, out_h, out_c = self.lstm(inpt, (h_0, c_0))
        output = self.label(out_h[-1])

        return output

