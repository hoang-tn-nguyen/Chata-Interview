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

class LSTM_Bi_Enc(nn.Module):
    def __init__(self, word_emb, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = word_emb
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, bidirectional=True, batch_first=True)
        self.fc_hid = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cel = nn.Linear(hid_dim * 2, hid_dim)
        
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

        hidden = hidden.view(-1, 2, hidden.shape[1], hidden.shape[2]).permute(0,2,1,3) # (N,B,D,E)
        cell = cell.view(-1, 2, cell.shape[1], cell.shape[2]).permute(0,2,1,3) # (N,B,D,E)
        hidden = hidden.reshape(hidden.shape[0], hidden.shape[1], -1) # (N,B,D*E)
        cell = cell.reshape(cell.shape[0], cell.shape[1], -1) # (N,B,D*E)
        
        hidden, cell = self.fc_hid(hidden), self.fc_cel(cell)
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
    def __init__(self, src_vocab_emb, tgt_vocab_emb, emb_dim, hid_dim, n_layers, dropout=0.1, bidirectional=True):
        super().__init__()
        if bidirectional:
            self.encoder = LSTM_Bi_Enc(src_vocab_emb, emb_dim, hid_dim, n_layers, dropout)
        else:
            self.encoder = LSTM_Enc(src_vocab_emb, emb_dim, hid_dim, n_layers, dropout)
        self.decoder = LSTM_Dec(tgt_vocab_emb, emb_dim, hid_dim, n_layers, dropout)
        
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

class PerceiverLayer(nn.Module): # A Perceiver-style encoder layer
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout):
        super().__init__()
        self.crs_att = MultiheadAttention(embed_dim, num_heads, dropout)
        self.crs_fwd = FeedForward(embed_dim, fwd_dim, dropout)
        self.slf_att = MultiheadAttention(embed_dim, num_heads, dropout)
        self.slf_fwd = FeedForward(embed_dim, fwd_dim, dropout)
        
    def forward(self, query, key, value, pad_mask=None, att_mask=None):
        output, heatmap = self.crs_att(query,key,value,pad_mask,att_mask)
        output = self.crs_fwd(output)
        output = self.slf_att(output,output,output)[0]
        output = self.slf_fwd(output)
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
            input = self.linear_layer(input) # (B,L,F) --> (B,L,E)
        input = self.posit_embeds(input) # (B,L,E)

        output = input # (B,L,E)
        for layer in self.transformer_layers:
            output, heatmap = layer(output, output, output, pad_mask, att_mask)
        return output # (B,L,E)

class Perceiver(nn.Module): # Perceiver encoder blocks
    def __init__(self, num_queries, num_layers, embed_dim, num_heads, fwd_dim, dropout, in_features=None, max_positions=10000, use_fourier=True):
        super().__init__()
        perceiver_layer_0 = PerceiverLayer(embed_dim,num_heads,fwd_dim,dropout) # avoid overfitting
        perceiver_layer_i = PerceiverLayer(embed_dim,num_heads,fwd_dim,dropout) # share params across all layers
        self.perceiver_layers = nn.ModuleList([
            perceiver_layer_i if i else perceiver_layer_0 
            for i in range(num_layers)])
        
        self.num_queries = num_queries
        self.query_embeds = nn.Embedding(self.num_queries, embed_dim)

        self.linear_layer = nn.Linear(in_features, embed_dim) if in_features != None else None
        self.posit_embeds = PositionalEncoding(max_positions, embed_dim, dropout, use_fourier)
        
    def forward(self, input, pad_mask=None, att_mask=None):
        if self.linear_layer != None:
            input = self.linear_layer(input) # (B,K,F) --> (B,K,E)
        input = self.posit_embeds(input) # (B,K,E)

        qry_idx = torch.arange(self.num_queries).unsqueeze(0).repeat(input.shape[0],1).to(input.device) # (B,Q)
        qry_emb = self.query_embeds(qry_idx) # (B,Q,E)

        output = qry_emb # (B,Q,E)
        for layer in self.perceiver_layers:
            output, heatmap = layer(output, input, input, pad_mask, att_mask)
        return output # (B,Q,E)

class PerceiverIO(nn.Module): # Perceiver model with decoder
    def __init__(self, num_outputs, num_queries, num_layers, embed_dim, num_heads, fwd_dim, dropout, in_features=None, out_features=None, max_positions=10000, use_fourier=True):
        super().__init__()
        self.perceiver = Perceiver(num_queries, num_layers, embed_dim, num_heads, fwd_dim, dropout, in_features, max_positions, use_fourier)
        
        self.out_att = MultiheadAttention(embed_dim, num_heads, dropout)
        self.out_fwd = FeedForward(embed_dim, fwd_dim, dropout)
        self.out_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim), # Remeber to normalize the output!
            nn.Linear(embed_dim, out_features),
        )

        self.num_outputs = num_outputs
        self.out_embeds = nn.Embedding(num_outputs, embed_dim)
        
    def forward(self, input, pad_mask=None, att_mask=None):
        input = self.perceiver(input, pad_mask, att_mask) # (B,Q,E)
        
        out_idx = torch.arange(self.num_outputs).unsqueeze(0).repeat(input.shape[0],1).to(input.device) # (B,O)
        out_emb = self.out_embeds(out_idx) # (B,O,E)
        
        output, heatmap = self.out_att(out_emb, input, input, pad_mask, att_mask, with_skip_connection=False) # (B,O,E)
        output = self.out_fwd(output) # (B,O,E)
        output = self.out_mlp(output) # (B,O,C)
        return output.squeeze(1) # (B,O,C) or (B,C) if O == 1

class Performer(nn.Module): # PERceiver encoder + TransFORMER decoder
    def __init__(self, perceiver, embed_dim, num_heads, fwd_dim, dropout, num_layers, max_positions=10000, use_fourier=True):
        super().__init__()
        self.perceiver = perceiver
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, fwd_dim, dropout)
            for _ in range(num_layers)
        ])
        self.posit_embeds = PositionalEncoding(max_positions, embed_dim, dropout, use_fourier)

    def forward(self, input, output, in_pad_mask=None, in_att_mask=None, out_pad_mask=None, out_att_mask=None, input_to_perceiver=True):
        if input_to_perceiver:
            input = self.perceiver(input, in_pad_mask, in_att_mask) # (B,Q,E)
        output = self.posit_embeds(output) # (B,O,E)

        for layer in self.decoder_layers:
            output, heatmap = layer(output,input,input,out_pad_mask,out_att_mask)
        return output # (B,O,E)

class TextGenerator(nn.Module):
    def __init__(self, performer, embed_dim, word_embed=None):
        super().__init__()
        self.performer = performer # The Perceiver input + Transformer output
        self.embedding = word_embed # Turn input sequence (B,*) to (B,*,E)
        self.prediction = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.embedding.size),
            nn.Softmax(dim=-1)
        )

    def forward(self, input, output=None, in_pad_mask=None, s_id=1, e_id=2, p_id=3, max_len=100, input_to_perceiver=True):
        '''
        input (B,I,E): Input sequence
        output (B,O): Output sequence
        '''
        if output == None:            
            return self.generate(input, in_pad_mask, s_id, e_id, p_id, max_len)
        else:
            out_pad_mask = (output == p_id) # (B,O)
            out_att_mask = self.generate_square_subsequent_mask(output.shape[1]).to(output.device) # (O,O)
            output = self.embedding(output) # (B,O) --> (B,O,E)

            output = self.performer(
                input, # (B,I,E)
                output, # (B,O,E)
                in_pad_mask = in_pad_mask,
                out_pad_mask = out_pad_mask,
                out_att_mask = out_att_mask,
                input_to_perceiver = input_to_perceiver,
            ) # (B,O,E)

            output = self.prediction(output) # (B,O,C)
            return output # (B,O,C)

    def generate(self, input, in_pad_mask=None, s_id=1, e_id=2, p_id=3, max_len=100):
        '''
        input (B,I,E): Embedded input sequence \n
        in_pad_mask (B,I): Pad mask for input sequence \n
        '''
        input = self.performer.perceiver(input, pad_mask=in_pad_mask) # (B,I,E)
        output = torch.full((input.shape[0],1), fill_value=s_id, dtype=torch.long, device=input.device) # (B,1)
        for _ in range(1,max_len):
            prob = self.forward(input, output, in_pad_mask, s_id, e_id, p_id, max_len, False) # (B,O,C)
            amax = prob[:,-1,:].max(dim=-1)[1].unsqueeze(1) # (B,1)
            output = torch.cat([output, amax], dim=-1) # (B,O+1)
        return output # (B,O)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class MachineTranslationModel(nn.Module):
    def __init__(self, src_vocab_emb, tgt_vocab_emb, latent_embed, embed_dim, ffwd_dim, num_heads, num_enc_layers, num_dec_layers, dropout):
        super().__init__()
        self.embedding = src_vocab_emb
        perceiver = Perceiver(latent_embed, num_enc_layers, embed_dim, num_heads, ffwd_dim, dropout)
        performer = Performer(perceiver, embed_dim, num_heads, ffwd_dim, dropout, num_dec_layers)
        self.generator = TextGenerator(performer, embed_dim, tgt_vocab_emb)

    def forward(self, input, output=None, in_pad_mask=None, s_id=1, e_id=2, p_id=3, max_len=100):
        in_pad_mask = (input == p_id)
        input = self.embedding(input)
        return self.generator(input, output, in_pad_mask, s_id, e_id, p_id, max_len)

if __name__ == '__main__':
    enc = WordEmbedding(1000, 128)
    inp = torch.randint(0,1000,(8,100))
    out = torch.randint(0,1000,(8,30))
    model = LSTM_ED(enc, enc, 128, 128, 4, 0.2, bidirectional=True)
    model(inp).shape

    enc = WordEmbedding(1000, 128)
    dec = WordEmbedding(1000, 128)
    model = MachineTranslationModel(enc, dec, 100, 128, 256, 1, 6, 6, 0.1)
    model(inp).shape


