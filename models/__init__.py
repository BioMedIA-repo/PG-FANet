import numpy as np

np.random.seed(0)

from models.seg_models.PG_FANet import *
MODELS = {
    'PGFANet': PG_FANet,
}
