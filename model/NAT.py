import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_
import numpy as np

from IPython import embed

from .encoder import EncoderStack
from .decoder import DecoderStack, PositionalEncoding
from .fertility_predictor import FertilityPredictor
from .translation_predictor import TranslationPredictor

class NAT(nn.Module):
    """Non-Autoregressive Transformer.
    
    Args:
        vocab_src: int, the size of source vocabulary.
        vocab_tgt: int, the size of target vocabulary.
        d_embed: int, the dimension of embedded input.
        S: int, the length of source input sentence.
        L: int, the number of the classes used to represent fertility.
        
    Shape:
        input: LongTensor, [N, S]
        output: FloatTensor, [T, N, vocab_tgt], T=S*L    
    """
    def __init__(self, vocab_src, vocab_tgt, S, d_embed=512, L=50, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(NAT, self).__init__()
        
        self.d_embed = d_embed
        self.S = S
        self.L = L
        self.T = S*L
        max_seq_len = S*L
        
        self.embedding_input = nn.Embedding(vocab_src, d_embed)
        self.position_encoder_en = PositionalEncoding(d_embed, S)
        self.position_encoder_de = PositionalEncoding(d_embed, S*L)
        self.encoder = EncoderStack(d_embed, nhead, num_encoder_layers, dim_feedforward, dropout, activation)
        self.fertility_predictor = FertilityPredictor(d_embed, L)
        self.decoder = DecoderStack(d_embed, nhead, max_seq_len, num_decoder_layers, dim_feedforward, dropout, activation)
        self.translation_predictor = TranslationPredictor(d_embed, vocab_tgt)

        self._reset_parameters()

    def forward(self, input):

        
        input_e = self.embedding_input(input.transpose(0,1)) # input_e: [S, N, E]
        input_pe = self.position_encoder_en(input_e) # input_pe: [S, N, E]
        encoder_output = self.encoder(input_pe) # encoder_output: [S, N, E]
        fertility_list = self.fertility_predictor(encoder_output) # fertility_list: [S, N]
        copied_embedding = self.copy_fertility(input_e, fertility_list, self.L) # copied_embedding: [T, N, E]
        copied_embedding_pe = self.position_encoder_de(copied_embedding) # copied_embedding_pe: [T, N, E]
        memory = encoder_output

        # embed()

        decoder_output = self.decoder(copied_embedding_pe, memory) # decoder_output: [T, N, E]
        output = self.translation_predictor(decoder_output) # output: [T, N, E]
        
        return output
        
        
    def copy_fertility(self, input_e, fertility_list, L):
        """Copy the input embedding as the number at corresponding index.
        
        Args:
            input_e: [S, N, E].
            fertility_list: [S, N].
            L: int, the number of the classes used to represent fertility.
        
        Output:
            copied_embedding: [T, N, E]
        """
        # copy as fertitlity list
        [S, N, E] = input_e.shape
        copied_embedding = torch.zeros(N, S*L, E) # copied_embedding: [N, S*L, E]
        input_e_permute = input_e.permute(1,0,2) # input_e_permute: [N, S, E]
        fertility_list_permute = fertility_list.transpose(0,1) # fertility_list_permuet: [N, S]
        
        # use fertility list and embedded input to get decoder's input.
        for i, fertility_batch in enumerate(fertility_list_permute):
            pos = 0
            for j, fertility_j in enumerate(fertility_batch):
                if fertility_j == 0:
                    continue
                copied_embedding[i,pos:pos+int(fertility_j),:] = input_e_permute[i,j,:].repeat(1,int(fertility_j),1)
                pos += int(fertility_j)
        copied_embedding = copied_embedding.transpose(0,1)
        
        return copied_embedding

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)