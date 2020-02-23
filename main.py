import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import math
import torch
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

input = torch.randint(0, 1000, (16, 20)).long().cuda() # input: [N, S]
model = NAT(vocab_src=1000, vocab_tgt=500, S=20).cuda()

output = model(input)

print(output.size())