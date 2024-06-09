import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentRNN(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, output_dim, embedding_dim, drop_prob=0.5):
        super(SentimentRNN, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        self.vocab_size = vocab_size

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # lstm
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=no_layers,
            dropout=drop_prob,
            bidirectional=True,
            batch_first=True
        )

        # dropout layer
        # self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim * 2, output_dim)
        self.sig = nn.Sigmoid()

    def representation(self,x):
        x=self.embedding(x)
        lstm_out, _ = self.lstm(x)
        hidden_out = torch.cat((lstm_out[:, -1, :self.hidden_dim], lstm_out[:, 0, self.hidden_dim:]), dim=1)
        return hidden_out

    def forward(self,x):
        x=self.embedding(x)
        lstm_out, _ = self.lstm(x)
        hidden_out = torch.cat((lstm_out[:, -1, :self.hidden_dim], lstm_out[:, 0, self.hidden_dim:]), dim=1)
        out = self.fc(hidden_out)
        # sig_out = self.sig(out)
        sig_out=F.softmax(out,dim=1)
        if len(sig_out.shape)==1:
            sig_out=sig_out.view(-1,sig_out.size(0))
        return sig_out

    # def forward(self, x, hidden):
    #     batch_size = x.size(0)
    #     # embeddings and lstm_out
    #     embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
    #     # print(embeds.shape)  #[50, 500, 1000]
    #     lstm_out, hidden = self.lstm(embeds, hidden)
    #     lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim * 2)
    #     # dropout and fully connected layer
    #     # out = self.dropout(lstm_out)
    #     out = self.fc(lstm_out)
    #     # sigmoid function
    #     sig_out = self.sig(out)
    #     # reshape to be batch_size first
    #     sig_out = sig_out.view(batch_size, -1)
    #     sig_out = sig_out[:, -1]  # get last batch of labels
    #     # return last sigmoid output and hidden state
    #     return sig_out, hidden

    # def init_hidden(self, batch_size):
    #     ''' Initializes hidden state '''
    #     # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
    #     # initialized to zero, for hidden state and cell state of LSTM
    #     # h0 = torch.zeros((self.no_layers * 2, batch_size, self.hidden_dim)).to(device)
    #     # c0 = torch.zeros((self.no_layers * 2, batch_size, self.hidden_dim)).to(device)
    #     h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
    #     c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
    #     hidden = (h0, c0)
    #     return hidden

def bilstm():
    return SentimentRNN(2, 1001, 64, 2, 64,0.5)