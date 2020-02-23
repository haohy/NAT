import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class TranslationPredictor(nn.Module):
    """The translation predictor of NAT.

    Args:
        d_embed: the dimension of EncoderStack's output.
        vocab: the number of the classes used to represent fertility.
    """
    def __init__(self, d_embed, vocab):
        super(TranslationPredictor, self).__init__()
        
        self.fc_layer = nn.Linear(d_embed, vocab)
        self.relu = nn.ReLU()
        
    def forward(self, decoder_output):
        """Using DecoderStack's output to predict translation.
        
        Shape:
            decoder_output: [S, N, E]
            translation_output: [S, N, vocab]
        """
        translation_output = F.softmax(self.relu(self.fc_layer(decoder_output)), dim=-1)
        
        return translation_output