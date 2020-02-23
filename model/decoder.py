import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class PositionalEncoding(nn.Module):
    """Positional Encoding.
    
    Args:
        d_embed: int, the dimension of embedding.
        max_seq_len: int, the max length of what to be summed with positional encoding.
    
    """
    def  __init__(self, d_embed, max_seq_len):
        super(PositionalEncoding, self).__init__()
        
        self.d_embed = d_embed
        pe = torch.zeros(max_seq_len, d_embed) # pe: [T, E]
        for pos in range(max_seq_len):
            for i in range(0, d_embed, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_embed)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_embed)))
                
        pe = pe.unsqueeze(0) # pe: [1, T, E]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_embed) # x: [T, N, E]
        #add constant to embedding
        seq_len = x.size(0)
        
        pe = self.pe[:, :seq_len].repeat(x.size(1),1,1).transpose(0,1) # pe: [T, N, E]
        pe = Variable(pe, requires_grad=False)
        x = x.cuda() if pe.is_cuda else x
        x = x + pe
        
        return x

class MultiHeadPositionalAttention(nn.Module):
    """Multi-Head Positional Attention sublayer."""
    def __init__(self, d_embed, num_heads, max_seq_len=100, dropout=0., bias=True, add_bias_kv=False,
                 add_zero_attn=False, kdim=None, vdim=None):
        super(MultiHeadPositionalAttention, self).__init__()
        
        self.positional_encoding = PositionalEncoding(d_embed, max_seq_len)
    
        self.multi_head_attn = nn.MultiheadAttention(d_embed, num_heads, dropout, bias, 
                                                     add_bias_kv, add_zero_attn, kdim, vdim)
        
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, 
                attn_mask=None):
        query = self.positional_encoding(query) # query: [L, N, E]
        key = self.positional_encoding(key) # key: [S, N, E]
        value = self.positional_encoding(value) # value: [S, N, E], L = S
        
        # attn_output: [L, N, E], attn_output_weights: [N, L, S]
        attn_output, _ = self.multi_head_attn(query, key, value, key_padding_mask, 
                                                                need_weights, attn_mask)
        
        return attn_output.contiguous()


class DecoderLayer(nn.Module):
    """The sublayer of DecoderStack.
    """
    def __init__(self, d_embed, nhead, max_seq_len=100, dim_feedforward=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.multi_head_self_attn = nn.MultiheadAttention(d_embed, nhead, dropout)
        self.multi_head_pos_attn = MultiHeadPositionalAttention(d_embed, nhead, max_seq_len, dropout)
        self.multi_head_inter_attn = nn.MultiheadAttention(d_embed, nhead, dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_embed, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_embed)

        # Implementation of MultiHead-Attentions
        self.norm1 = nn.LayerNorm(d_embed)
        self.norm2 = nn.LayerNorm(d_embed)
        self.norm3 = nn.LayerNorm(d_embed)
        self.norm4 = nn.LayerNorm(d_embed)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.d_embed = d_embed
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """DecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
        
        Shape:
            tgt: [T, N, E].
            memory: [S, N, E]
        """
        tgt_len, bsz = tgt.size(0), tgt.size(1)
        
        # define tgt_mask
        if tgt_mask is None:
            diag_ones = np.array([1]*tgt_len)
            tgt_mask = torch.from_numpy(np.diag(diag_ones)).bool()
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask, float('-inf'))
        
        # multi-head self-attention
        tgt2 = self.multi_head_self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # multi-head positional attention
        tgt2 = self.multi_head_pos_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # multi-head inter-attention
        tgt2 = self.multi_head_inter_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        # position-wise feed forward layer
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        
        return tgt.contiguous()


class DecoderStack(nn.Module):
    """The decoder stack of NAT.
    """
    def __init__(self, d_embed=512, nhead=8, max_seq_len=1000, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(DecoderStack, self).__init__()
        
        decoder_layer = DecoderLayer(d_embed, nhead, max_seq_len, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_embed)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        Shape:
            tgt: [T, N, E]
            memory: [S, N, E]
            output: [T, N, E]
        """
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        
        return output.contiguous()