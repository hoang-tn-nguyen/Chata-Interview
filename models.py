import torch
import torch.nn as nn
import math

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.voc_emb = nn.Embedding(vocab_size, embed_dim)
        self.size = vocab_size

    def forward(self, input):
        output = self.voc_emb(input) # (B,L,E)
        return self.dropout(output)

# --- LSTM Baseline ---
class LSTM_Enc(nn.Module):
    def __init__(self, word_emb, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = word_emb
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input):
        '''
        input: (B,L) \n
        --- \n
        outputs: (B,L,E*D) \n
        hidden: (N*D,B,E) \n
        cell: (N*D,B,E) \n
        where N is num_layers, D is num_directions \n
        '''
        input = self.embedding(input)
        outputs, (hidden, cell) = self.rnn(input)
        return hidden, cell

class LSTM_Dec(nn.Module):
    def __init__(self, word_emb, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = word_emb
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)    
        self.dropout = nn.Dropout(dropout)
        self.prediction = nn.Sequential(
            nn.Linear(in_features=hid_dim, out_features=self.embedding.size),
            nn.Softmax(dim=-1),
        )

    def forward(self, hidden, cell, output=None, max_len=100, sid=1):
        '''
        hidden: (N*D,B,E) \n
        cell: (N*D,B,E) \n
        output: (B,L) \n
        where N is num_layers, D is num_directions \n
        '''
        if output == None:
            bsz = hidden.shape[1]
            output = torch.full((bsz, 1), fill_value=sid, device=hidden.device)
            
            for i in range(1,max_len):
                out_emb, (hidden, cell) = self.rnn(self.embedding(output[:,-1].unsqueeze(-1)), (hidden, cell))
                next_output = self.prediction(out_emb).max(dim=-1)[1]
                output = torch.cat([output, next_output], dim=1)
        else:
            output = self.embedding(output)
            output, (hidden, cell) = self.rnn(output, (hidden, cell))     
            output = self.prediction(output)   
        return output

class LSTM_ED(nn.Module):
    def __init__(self, word_emb, emb_dim, hid_dim, n_layers, dropout=0.1):
        super().__init__()
        self.encoder = LSTM_Enc(word_emb, emb_dim, hid_dim, n_layers, dropout)
        self.decoder = LSTM_Dec(word_emb, emb_dim, hid_dim, n_layers, dropout)
        
    def forward(self, input, output=None, input_len=None):
        hidden, cell = self.encoder(input)
        output = self.decoder(hidden, cell, output)
        return output

# --- Transformer Baseline ---
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

class FeedForward(nn.Module):
    def __init__(self, embed_dim, fwd_dim, dropout):
        super().__init__()
        self.fwd_layer = nn.Sequential(
            nn.Linear(embed_dim, fwd_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fwd_dim, embed_dim))
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, input, with_skip_connection=True):
        output = self.fwd_layer(self.normalize(input))
        if with_skip_connection:
            output = output + input
        return output # (B,S,E)

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout)
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, pad_mask=None, att_mask=None, with_skip_connection=True):
        norm_query = self.normalize(query) # (B,Q,E)
        norm_key = self.normalize(key) # (B,K,E)
        norm_value = self.normalize(value) # (B,V,E)
        
        output, heatmap = self.attention(
            norm_query.permute(1,0,2), 
            norm_key.permute(1,0,2), 
            norm_value.permute(1,0,2),
            key_padding_mask=pad_mask, 
            attn_mask=att_mask) # (Q,B,E), (B,Q,K)
        output = output.permute(1,0,2) # (B,Q,E)

        if with_skip_connection:
            output = output + query # (B,Q,E)
        return output, heatmap # (B,Q,E), (B,Q,K)

class TransformerLayer(nn.Module): # A GPT-2 style encoder/decoder layer
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout):
        super().__init__()
        self.crs_att = MultiheadAttention(embed_dim, num_heads, dropout)
        self.crs_fwd = FeedForward(embed_dim, fwd_dim, dropout)
        
    def forward(self, query, key, value, pad_mask=None, att_mask=None):
        output, heatmap = self.crs_att(query,key,value,pad_mask,att_mask)
        output = self.crs_fwd(output)
        return output, heatmap

class TransformerDecoderLayer(nn.Module): # A Transformer-style decoder layer
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout):
        super().__init__()
        self.slf_att = MultiheadAttention(embed_dim, num_heads, dropout)
        self.crs_att = MultiheadAttention(embed_dim, num_heads, dropout)
        self.crs_fwd = FeedForward(embed_dim, fwd_dim, dropout)

    def forward(self, query, key, value, pad_mask=None, att_mask=None):
        query, heatmap = self.slf_att(query, query, query, pad_mask, att_mask) # (B,O,E)
        output, heatmap = self.crs_att(query, key, value) # (B,O,E)
        output = self.crs_fwd(output) # (B,O,E)
        return output, heatmap # (B,O,E)

class Transformer(nn.Module): # GPT-2 blocks
    def __init__(self, num_layers, embed_dim, num_heads, fwd_dim, dropout, in_features=None, max_positions=1000, use_fourier=True):
        super().__init__()
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim,num_heads,fwd_dim,dropout) 
            for _ in range(num_layers)])
        
        self.linear_layer = nn.Linear(in_features, embed_dim) if in_features != None else None
        self.posit_embeds = PositionalEncoding(max_positions, embed_dim, dropout, use_fourier)
        
    def forward(self, input, pad_mask=None, att_mask=None):
        if self.linear_layer != None:
            input = self.linear_layer(input) # (B,K,F) --> (B,K,E)
        input = self.posit_embeds(input) # (B,K,E)

        output = input # (B,Q,E)
        for layer in self.transformer_layers:
            output, heatmap = layer(output, output, output, pad_mask, att_mask)
        return output # (B,Q,E)

if __name__ == '__main__':
    enc = WordEmbedding(1000, 128)
    inp = torch.randint(0,1000,(8,100))
    out = torch.randint(0,1000,(8,30))
    model = LSTM_ED(enc, 128, 128, 1, 0.0)
    model(inp).shape
