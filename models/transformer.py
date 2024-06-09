import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, num_classes, max_seq_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = nn.Embedding(max_seq_len, embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout=0.1,batch_first=True),
            num_layers,
        )
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def representation(self, x):
        seq_len, batch_size = x.size()
        positions = torch.arange(0, seq_len).unsqueeze(1).expand(seq_len, batch_size).to(x.device)
        x = self.embedding(x) + self.pos_encoder(positions)
        x = self.transformer_encoder(x)
        x=self.pool(x.permute(0, 2, 1)).squeeze()
        return x

    def get_grandiants(self,x):
        seq_len, batch_size = x.size()
        positions = torch.arange(0, seq_len).unsqueeze(1).expand(seq_len, batch_size).to(x.device)
        x = self.embedding(x)
        y=self.pos_encoder(positions)
        x=x+y
        return x

    def predict(self,x):
        # x = self.transformer_encoder(x)
        # x = torch.avg(x, dim=0)  # Pooling
        x = self.transformer_encoder(x)
        x = self.pool(x.permute(0, 2, 1)).squeeze()
        x = self.fc(x)
        x = x.view(-1, x.size(0))
        x = F.softmax(x,dim=1)
        return x

    def forward(self, x):
        seq_len, batch_size = x.size()
        positions = torch.arange(0, seq_len).unsqueeze(1).expand(seq_len, batch_size).to(x.device)
        x = self.embedding(x)
        y=self.pos_encoder(positions)
        x=x+y
        x = self.transformer_encoder(x)
        # x = torch.avg(x, dim=0)  # Pooling
        x=self.pool(x.permute(0, 2, 1)).squeeze()
        x = self.fc(x)
        if seq_len==1:
            x= x.view(-1,x.size(0))
        x= F.softmax(x,dim=1)
        return x

def transformer():
    return TransformerModel(1001, 64, 8, 3, 256, 2, 500)