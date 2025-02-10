import random, os
import numpy as np
import torch
seed = 55
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
import pandas as pd
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'
device = 'cpu'
import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import pandas as pd
import dgl
from dgl.data import DGLDataset
from dgl.data import Subset
from random import shuffle
from dgl import RemoveSelfLoop
from random import sample
from dgl import transforms as T

import dgl
import dgl.nn.pytorch as dglnn
from torch_geometric.utils import coalesce, scatter, softmax
from dgl import ToSimple
import copy