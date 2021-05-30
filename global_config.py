import os
import torch

here = os.path.abspath(os.sep.join(__file__.split(os.sep)[:-1]))
DATA_DIR = os.path.join(here, 'data')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_LENGTH = 20