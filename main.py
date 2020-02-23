import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchtext.data import Field, BucketIterator
import numpy as np

import spacy
import en_core_web_sm
import de_core_news_sm
spacy_en = en_core_web_sm.load()
spacy_de = de_core_news_sm.load()

from IPython import embed

from model import NAT

input = torch.randint(0, 100, (16, 20)).long() # input: [N, S]

# torch.cuda.empty_cache()

# vocab_src, vocab_tgt, S, d_embed=512, L=50, nhead=8, num_encoder_layers=6,
#                  num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu"

model = NAT(vocab_src=100, vocab_tgt=50, S=20, num_encoder_layers=4, num_decoder_layers=4,\
    dim_feedforward=512)
num_parameters_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("the number of parameters = {}".format(num_parameters_train))

# model = nn.DataParallel(model)

output = model(input)


print(output.size())