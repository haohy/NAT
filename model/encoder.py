import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class EncoderStack(nn.Module):
    """The encoder stack of NAT.
    """
    def __init__(self, d_embed=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(EncoderStack, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_embed, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_embed)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Shape:
            src: [S, N, E]
            output: [S, N, E]
        """
        output = self.encoder(src)

        return output
