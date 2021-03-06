
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F 

class LSTM_Classifier(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(LSTM_Classifier, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):

        batch_size=x.size(0)

        embeds=self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out=lstm_out.contiguous().view(-1, self.hidden_dim)
        out=self.dropout(lstm_out)
        out=self.fc(out)
        sig_out=self.sigmoid(out)

        sig_out=sig_out.view(batch_size, -1)
        sig_out=sig_out[:, -1]

        return sig_out, hidden

    def init_hidden(self, batch_size):

        weight=next(self.parameters()).data

        train_on_gpu=torch.cuda.is_available()

        if train_on_gpu:

            hidden=(weight.new(self.n_layers, batch_size,
            self.hidden_dim).zero_().cuda(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())

        else:
            hidden=(weight.new(self.n_layers, batch_size,
            self.hidden_dim).zero_(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden    




