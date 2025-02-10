import torch
import time
from compress.datasets import ImageFolder
import numpy as np
from compressai.zoo import *

from compress.utils.annealings import *
from compress.utils.stf.loop import train_one_epoch, configure_optimizers
from compress.utils.stf.parser import parse_args
from compress.utils.stf.kodak import TestKodakDataset
from compress.utils.stf.plotting import plot_sos, plot_rate_distorsion
from compress.utils.stf.utils import *
from compress.utils.stf.loading import *
torch.backends.cudnn.benchmark = True #sss
import wandb











image_models = {"zou22-base": aux_net_models["stf"],
                "zou22-sos":models["cnn_multi"],

                }















