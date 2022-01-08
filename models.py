import torch
import torch.nn as nn
import math

# Need to use data augmentation, transfer learning
# Need to test model carefully using different validation techniques.
# They mentioned something about (LSTM, CNN, RNN, seq2seq, BERT etc.)

# Try LSTM to transform a sequence to another sequence.
# Try Transformer.
# Use the two models as baselines, try to find other papers that work on this dataset, see how good they perform. Compare with them.
# Probably try to improve the model with something better.

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim, dropout=0.0, use_fourier=True):
        super().__init__()
        assert embed_dim % 2 == 0
        self.dropout = nn.Dropout(dropout)
        self.use_fourier = use_fourier
        
        if use_fourier:
            position = torch.arange(max_len).unsqueeze(1) 
            div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
            self.pos_enc = torch.zeros(1, max_len, embed_dim)
            self.pos_enc[0, :, 0::2] = torch.sin(position * div_term)
            self.pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        else:
            self.pos_enc = nn.Embedding(max_len, embed_dim)

    def forward(self, input):
        if self.use_fourier:
            input = input + self.pos_enc[:,:input.shape[1]].to(input.device)
        else:
            pos_idx = torch.arange(input.shape[1]).unsqueeze(0).repeat(input.shape[0],1).to(input.device) # (B,K)
            pos_emb = self.pos_enc(pos_idx) # (B,K,E)
            input = input + pos_emb # (B,K,E)
        return self.dropout(input)

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.voc_emb = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, input):
        output = self.voc_emb(input) # (B,L,E)
        return self.dropout(output)

# --- LSTM Baseline ---
class LSTM_Enc(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input):
        '''
        input: (B,L,E) \n
        --- \n
        outputs: (B,L,N*E) \n

        
        '''
        outputs, (hidden, cell) = self.rnn(input)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        return hidden, cell

class LSTM_Dec(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)    
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden, cell, input=None):
        '''
        hidden:

        '''
        output, (hidden, cell) = self.rnn(input, (hidden, cell))        
        return output

class LSTM_ED(nn.Module):
    def __init__(self, word_emb):
        super().__init__()
        self.embedding = word_emb
        self.encoder = LSTM_Enc()
        self.decoder = LSTM_Dec()

    def forward(self, input, input_len=None):
        out = self.embedding(input) # (B,L,E)
        out = self.encoder(out)
        out = self.decoder(out)
        return out

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, output=None):
        pass

if __name__ == '__main__':
    enc = WordEmbedding(1000, 128)
    enc(torch.randint(0,1000,(8,100))).shape
