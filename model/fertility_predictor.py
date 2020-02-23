import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class FertilityPredictor(nn.Module):
    """The fertility predictor of NAT.
    
    Args:
        d_embed: the dimension of EncoderStack's output.
        L: the number of the classes used to represent fertility.
    """
    def __init__(self, d_embed, L):
        super(FertilityPredictor, self).__init__()
        
        self.fc_layer = nn.Linear(d_embed, L)
        self.relu = nn.ReLU()
        
    def forward(self, encoder_output):
        """Using EncoderStack's output to predict fertility list.
        
        Shape:
            encoder_output: [S, N, E]
            fertility_list: [S, N]
        """
        fertility_list = F.softmax(self.relu(self.fc_layer(encoder_output)), dim=-1) # fertility_list: [S, N, L]
        fertility_list = torch.argmax(fertility_list, dim=-1) # fertility_list: [S, N]
        
        return fertility_list