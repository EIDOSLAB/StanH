import torch 
import os 
import numpy as np 
from pathlib import Path
import random
from compress.utils.help_function import compute_msssim, compute_psnr
from torchvision import transforms
from PIL import Image
import torch
import time
from compress.datasets import ImageFolder
import torch.nn.functional as F
import math
from compressai.ops import compute_padding
import math 
from pytorch_msssim import ms_ssim
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch.optim as optim
import argparse
from compressai.zoo import *
from torch.utils.data import DataLoader
from os.path import join 
from compress.zoo import models, aux_net_models
import wandb
from torch.utils.data import Dataset
from os import listdir
from collections import OrderedDict
from compress.utils.annealings import *
from compress.quantization.activation import NonLinearStanh
torch.backends.cudnn.benchmark = True #sss



import torch.nn as nn
from pytorch_msssim import ms_ssim 

from datetime import datetime
from os.path import join 
import wandb
import shutil


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("-m","--model",default="0728_last_.pth.tar",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-mp","--model_path",default="/scratch/inference/new_models/devil2022/rebuttal_model/A40/raws",help="Model architecture (default: %(default)s)",)
    
    
    parser.add_argument("--lmbda", nargs='+', type=float, default =[0.007,0.0095,0.015])
    
    parser.add_argument("-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",)

    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    parser.add_argument("-ni","--num_images",default = 24064, type = int)
    parser.add_argument("-niv","--num_images_val",default = 1024, type = int)

    parser.add_argument("-sp","--stanh_path",default="/scratch/inference/new_models/devil2022/4_anchors_stanh",help="Model architecture (default: %(default)s)",)#dddd
    parser.add_argument("-rp","--result_path",default="/scratch/inference/results",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-ip","--image_path",default="/scratch/dataset/kodak",help="Model architecture (default: %(default)s)",)
    parser.add_argument("--entropy_estimation", action="store_true", help="Use cuda")
    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--pretrained_stanh", action="store_true", help="Use cuda")
    parser.add_argument("--only_dist", action="store_true", help="Use cuda")
    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--filename",default="/data/",type=str,help="factorized_annealing",)
    
    parser.add_argument("-e","--epochs",default=600,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument("--fact_gp",default=15,type=int,help="factorized_beta",)
    parser.add_argument("--gauss_gp",default=15,type=int,help="gauss_beta",)
    parser.add_argument("--gauss_tr",default=True,type=bool,help="gauss_tr",)
    parser.add_argument("--fact_annealing",default="gap_stoc",type=str,help="factorized_annealing",)
    parser.add_argument("--gauss_annealing",default="gap_stoc",type=str,help="factorized_annealing",)
    
    parser.add_argument("--num_stanh", type=int, default=3, help="Batch size (default: %(default)s)")
    parser.add_argument("--training_focus",default="stanh_levels",type=str,help="factorized_annealing",)

    args = parser.parse_args(argv) ###s
    return args

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def update_checkpopoint(state_dict,num_stanh):

    res =  OrderedDict()


    for k,v in state_dict.items():
        if "gaussian_conditional.0." in k:

            adding = str(6) 
            new_text = k.replace("gaussian_conditional.0.", "gaussian_conditional." + adding + ".")
            res[new_text] = state_dict[k]
            res[k] = state_dict[k] 

        elif "entropy_bottleneck.0." in k:

            adding = str(6) 
            new_text = k.replace("entropy_bottleneck.0.", "entropy_bottleneck." + adding + ".")
            res[new_text] = state_dict[k]
            res[k] = state_dict[k] 

        else:
            res[k]=state_dict[k]
    
    return res


def main():


    wandb.init(project="eval", entity="albipresta") 



    models_path = "/scratch/inference/new_models/devil2022/rebuttal_model/A40/raws/0728_last_.pth.tar" #ssss

    device = "cuda"
    checkpoint = torch.load(models_path, map_location=device)

    print("---> ",checkpoint["state_dict"].keys())








    factorized_configuration =checkpoint["factorized_configuration"]    
    gaussian_configuration =  checkpoint["gaussian_configuration"]#sssssdddd



    print("gaussian configuration: ",gaussian_configuration)


    factorized_configuration["beta"] = 10
    factorized_configuration["trainable"] = True
    factorized_configuration["annealing"] = "gap_stoc"
    factorized_configuration["gap_factor"] = 15


                
    gaussian_configuration["beta"] = 10
    gaussian_configuration["trainable"] = True
    gaussian_configuration["annealing"] = "gap_stoc"
    gaussian_configuration["gap_factor"] = 15



    factorized_configurations, gaussian_configurations = [],[]
    for jj in range(7) :

        factorized_configurations.append(factorized_configuration)
        gaussian_configurations.append(gaussian_configuration)


    # tirare fuori le stanh, intanto quelle delle derivation

    hyper_stanh_w, hyper_stanh_b = checkpoint["entropy_bottleneck_w"], checkpoint["entropy_bottleneck_b"]
    gauss_stanh_w, gauss_stanh_b = checkpoint["gaussian_conditional_w"], checkpoint["gaussian_conditional_b"]


    checkpoint["factorized_configuration"] = factorized_configurations
    checkpoint["gaussian_configuration"] = gaussian_configurations
    checkpoint["state_dict"] = update_checkpopoint(checkpoint["state_dict"],num_stanh = len(hyper_stanh_b) + 1) # dd one more stanh



    new_cheks = {}
    new_cheks["state_dict"] = checkpoint["state_dict"]
    new_cheks["factorized_configuration"] = factorized_configurations
    new_cheks["gaussian_configuration"] = gaussian_configurations
    torch.save(new_cheks, "/scratch/inference/new_models/devil2022/rebuttal_model/A40/model/0728_last_.pth.tar")




    for i, c in enumerate(hyper_stanh_w):

        name = "A040-D2" + str(i+1)  + ".pth.tar"
        filename = "/scratch/inference/new_models/devil2022/rebuttal_model/A40/stanh/"  + name  #dddd


        state_dict_stanh = {}

        state_dict_stanh["state_dict"] = {}

        state_dict_stanh["state_dict"]["gaussian_conditional"] = {}
        state_dict_stanh["state_dict"]["entropy_bottleneck"] = {}

        state_dict_stanh["state_dict"]["gaussian_conditional"]["w"] = gauss_stanh_w[i]
        state_dict_stanh["state_dict"]["gaussian_conditional"]["b"] = gauss_stanh_b[i]
        state_dict_stanh["state_dict"]["entropy_bottleneck"]["w"] = hyper_stanh_w[i]
        state_dict_stanh["state_dict"]["entropy_bottleneck"]["b"] = hyper_stanh_b[i]

        print("*********************************************************************")
        print("-----> ",i,":  ",c,"---> ",gauss_stanh_w[i].shape)
        print("-----> ",i,":  ",c,"---> ",gauss_stanh_b[i].shape)
        print("-----> ",i,":  ",c,"---> ",hyper_stanh_w[i].shape)
        print("-----> ",i,":  ",c,"---> ",hyper_stanh_b[i].shape)


        state_dict_stanh["factorized_configuration"] =  factorized_configurations[i]

        state_dict_stanh["gaussian_configuration"] = gaussian_configurations[i]

        torch.save(state_dict_stanh, filename)
        # stanh_checkpoints_p = args.stanh_path + "/anchors/q6-stanh.pth.tar"#a3-stanh.pth.tar" #q6-stanh.pth.tar"#a1-stanh.pth.tar"






    print("FATTO!!!!!")



if __name__ == "__main__":

    main()




                


