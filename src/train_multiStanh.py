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

torch.backends.cudnn.benchmark = True #sss



import torch.nn as nn
from pytorch_msssim import ms_ssim 

from datetime import datetime
from os.path import join 
import wandb
import shutil


def plot_sos(model, device,n = 1000, dim = 0,sl = 1 ):

    x_min = float((min(model.gaussian_conditional[sl].sos.b) + min(model.gaussian_conditional[sl].sos.b)*0.5).detach().cpu().numpy())
    x_max = float((max(model.gaussian_conditional[sl].sos.b)+ max(model.gaussian_conditional[sl].sos.b)*0.5).detach().cpu().numpy())
    step = (x_max-x_min)/n
    x_values = torch.arange(x_min, x_max, step)
    x_values = x_values.repeat(model.gaussian_conditional[sl].M,1,1)
            
    y_values=model.gaussian_conditional[sl].sos(x_values.to(device))[0,0,:]
    data = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
    table = wandb.Table(data=data, columns = ["x", "sos"])
    wandb.log({"GaussianSoS/Gaussian SoS at dimension " + str(dim): wandb.plot.line(table, "x", "sos", title='GaussianSoS/Gaussian SoS  with beta = {}'.format(model.gaussian_conditional.sos.beta))})
    y_values= model.gaussian_conditional[sl].sos(x_values.to(device), -1)[0,0,:]
    data_inf = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
    table_inf = wandb.Table(data=data_inf, columns = ["x", "sos"])
    wandb.log({"GaussianSoS/Gaussian SoS  inf at dimension " + str(dim): wandb.plot.line(table_inf, "x", "sos", title='GaussianSoS/Gaussian SoS  with beta = {}'.format(-1))})  






def save_checkpoint_our(state, is_best, filename,filename_best):
    torch.save(state, filename)
    wandb.save(filename)
    if is_best:
        shutil.copyfile(filename, filename_best)
        wandb.save(filename_best)



def create_savepath(args):
    now = datetime.now()
    date_time = now.strftime("%m%d")
    suffix = ".pth.tar"
    c = join(date_time,"last").replace("/","_")

    
    c_best = join(date_time,"best").replace("/","_")
    c = join(c,suffix).replace("/","_")
    c_best = join(c_best,suffix).replace("/","_")
    
    
    path = args.filename
    savepath = join(path,c)
    savepath_best = join(path,c_best)
    
    print("savepath: ",savepath)
    print("savepath best: ",savepath_best)
    return savepath, savepath_best



class RateDistortionLoss(nn.Module):

    def __init__(self, lmbda = 1e-2,  metric = "mse", only_dist = False ): #ddd
        super().__init__()


        if metric is "mse":
            self.dist_metric = nn.MSELoss()
        else:
            self.dist_metric = ms_ssim 
        self.lmbda = lmbda[0] if isinstance(lmbda,list) else lmbda
        self.only_dist = only_dist



    def forward(self, output, target, lmbda = None):
        N, _, H, W = target.size()      
        out = {}



        if self.dist_metric == ms_ssim:
            out["mse_loss"] = self.dist_metric(output["x_hat"], target, data_range=1)
            distortion = 1 - out["mse_loss"]
        else:
            out["mse_loss"] = self.dist_metric(output["x_hat"], target)
            distortion = 255**2 * out["mse_loss"]
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 


        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   
        

        lmbda = self.lmbda if lmbda is None else lmbda
        if self.only_dist is False:
            out["loss"] = lmbda * distortion + out["bpp_loss"] 
        else:
            out["loss"] = distortion 
        return out  
    






def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())
    optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)),lr=args.learning_rate,)
    return optimizer



def train_one_epoch(model, 
                    criterion, 
                    train_dataloader,
                      optimizer,
                        epoch, 
                        clip_max_norm ,
                        counter, 
                        annealing_strategy_entropybottleneck , 
                        annealing_strategy_gaussian,
                         lmbda_list = None ):
    model.train()
    device = next(model.parameters()).device

  
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    fact_beta = AverageMeter()
    gauss_beta = AverageMeter()


    for i, d in enumerate(train_dataloader):
        counter += 1
        d = d.to(device)

        optimizer.zero_grad()


        quality_index =  random.randint(0, model.num_stanh - 1)
        #quality_index =  1#random.randint(0, model.num_stanh - 1)
        out_net = model(d, training = True, stanh_level = quality_index)
        gap = out_net["gap"]

        out_criterion = criterion(out_net, d) if lmbda_list is None \
                                                else criterion(out_net, d, lmbda = lmbda_list[quality_index])
        out_criterion["loss"].backward()

        loss.update(out_criterion["loss"].clone().detach())
        mse_loss.update(out_criterion["mse_loss"].clone().detach())
        bpp_loss.update(out_criterion["bpp_loss"].clone().detach())


        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()



        if i % 10000 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                
                f'\tMSE loss: {out_criterion["mse_loss"].item() * 255 ** 2 / 3:.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'

            )



        wand_dict = {
            "train_batch": counter,
            #"train_batch/delta": model.gaussian_conditional.sos.delta.data.item(),
            "train_batch/losses_batch": out_criterion["loss"].clone().detach().item(),
            "train_batch/bpp_batch": out_criterion["bpp_loss"].clone().detach().item(),
            "train_batch/mse":out_criterion["mse_loss"].clone().detach().item(),
        }
        wandb.log(wand_dict)


        wand_dict = {
                "general_data/":counter,
                "general_data/factorized_gap: ": gap[0]
            }
            
        wandb.log(wand_dict)


        wand_dict = {
                    "general_data":counter,
                    "general_data/gaussian_gap: ": gap[1]
                }
                
        wandb.log(wand_dict)
            
        


        if  annealing_strategy_entropybottleneck is not  None:

            
            if annealing_strategy_entropybottleneck[quality_index].type == "triangle":
                annealing_strategy_entropybottleneck[quality_index].step(gap = gap[0])
                model.entropy_bottleneck[quality_index].sos.beta = annealing_strategy_entropybottleneck[quality_index].beta
            elif "random" in annealing_strategy_entropybottleneck[quality_index].type:
                annealing_strategy_entropybottleneck[quality_index].step(gap = gap[0])
                model.entropy_bottleneck[quality_index].sos.beta = annealing_strategy_entropybottleneck[quality_index].beta
        
            else:
                    
                lss = out_criterion["loss"].clone().detach().item()
                annealing_strategy_entropybottleneck[quality_index].step(gap[0], epoch, lss)
                model.entropy_bottleneck[quality_index].sos.beta = annealing_strategy_entropybottleneck[quality_index].beta

            fact_beta.update(annealing_strategy_entropybottleneck[quality_index].beta)

        if annealing_strategy_gaussian[quality_index] is not None:
            if annealing_strategy_gaussian[quality_index].type == "triangle":
                annealing_strategy_gaussian[quality_index].step(gap = gap[1])
                model.gaussian_conditional[quality_index].sos.beta = annealing_strategy_gaussian[quality_index].beta
            elif "random" in annealing_strategy_gaussian[quality_index].type:
                annealing_strategy_gaussian[quality_index].step(gap = gap[1])
                model.gaussian_conditional[quality_index].sos.beta = annealing_strategy_gaussian[quality_index].beta
            else:
                lss = out_criterion["loss"].clone().detach().item()
                annealing_strategy_gaussian[quality_index].step(gap[1], epoch, lss)
                model.gaussian_conditional[quality_index].sos.beta = annealing_strategy_gaussian[quality_index].beta


            wand_dict = {
                    "general_data/":counter,
                    "general_data/gaussian_beta: ": model.gaussian_conditional[quality_index].sos.beta
            }
                
            wandb.log(wand_dict)

            gauss_beta.update(model.gaussian_conditional[quality_index].sos.beta)

    log_dict = {
        "train":epoch,
        "train/loss": loss.avg,
        "train/bpp": bpp_loss.avg,
        "train/mse": mse_loss.avg,
        "train/gauss_beta": gauss_beta.avg
        }
        
    wandb.log(log_dict)
    return counter

import seaborn as sns
palette = sns.color_palette("tab10")
import matplotlib.pyplot as plt


def plot_rate_distorsion(bpp_res, psnr_res,epoch, eest = "compression", index_list = []):

    chiavi_da_mettere = list(psnr_res.keys())
    legenda = {}
    for i,c in enumerate(chiavi_da_mettere):
        legenda[c] = {}
        legenda[c]["colore"] = [palette[i],'-']
        legenda[c]["legends"] = c
        legenda[c]["symbols"] = ["*"]*300
        legenda[c]["markersize"] = [5]*300    

    plt.figure(figsize=(12,8)) # fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    list_names = list(psnr_res.keys())

    minimo_bpp, minimo_psnr = 10000,1000
    massimo_bpp, massimo_psnr = 0,0

    for _,type_name in enumerate(list_names): 

        bpp = bpp_res[type_name]
        psnr = psnr_res[type_name]
        colore = legenda[type_name]["colore"][0]
        #symbols = legenda[type_name]["symbols"]
        #markersize = legenda[type_name]["markersize"]
        leg = legenda[type_name]["legends"]

        bpp = torch.tensor(bpp).cpu()
        psnr = torch.tensor(psnr).cpu()    
        plt.plot(bpp,psnr,"-" ,color = colore, label =  leg ,markersize=8)       
        plt.plot(bpp, psnr, marker="o", markersize=4, color =  colore)


        for j in range(len(bpp)):
            if bpp[j] < minimo_bpp:
                minimo_bpp = bpp[j]
            if bpp[j] > massimo_bpp:
                massimo_bpp = bpp[j]
            
            if psnr[j] < minimo_psnr:
                minimo_psnr = psnr[j]
            if psnr[j] > massimo_psnr:
                massimo_psnr = psnr[j]

    minimo_psnr = int(minimo_psnr)
    massimo_psnr = int(massimo_psnr)
    psnr_tick =  [round(x) for x in range(minimo_psnr, massimo_psnr + 2)]
    plt.ylabel('PSNR', fontsize = 30)
    plt.yticks(psnr_tick)

    #print(minimo_bpp,"  ",massimo_bpp)

    bpp_tick =   [round(x)/10 for x in range(int(minimo_bpp*10), int(massimo_bpp*10 + 2))]
    plt.xticks(bpp_tick)
    plt.xlabel('Bit-rate [bpp]', fontsize = 30)
    plt.yticks(fontsize=27)
    plt.xticks(fontsize=27)
    plt.grid()

    plt.legend(loc='lower right', fontsize = 25)



    plt.grid(True)
    if eest == "model":
        wandb.log({"model":epoch,
              "model/rate distorsion trade-off": wandb.Image(plt)})
    else:  
        wandb.log({"compression":epoch,
              "compression/rate distorsion trade-off": wandb.Image(plt)})       
    plt.close()  
    print("FINITO")





def update_checkpopoint(state_dict,num_stanh):

    res =  OrderedDict()


    for k,v in state_dict.items():
        if "gaussian_conditional" in k:
            for j in range(num_stanh):
                adding = str(j) 
                new_text = k.replace("gaussian_conditional.", "gaussian_conditional." + adding + ".")
                res[new_text] = state_dict[k]
        elif "entropy_bottleneck" in k:
            for j in range(num_stanh):
                adding = str(j) 
                new_text = k.replace("entropy_bottleneck.", "entropy_bottleneck." + adding + ".")
                res[new_text] = state_dict[k]
        else:
            res[k]=state_dict[k]
    
    return res

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

image_models = {"zou22-base": aux_net_models["stf"],
                "zou22-sos":models["cnn_multi"],

                }




def rename_key(key):
    """Rename state_deeict key."""

    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]
    if key.startswith('h_s.'):
        return None

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    # if ".downsample." in key:
    #     return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key



def load_state_dict(state_dict):
    """Convert state_dict keys."""
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    if None in state_dict:
        state_dict.pop(None)
    return state_dict

def load_checkpoint(arch: str, checkpoint_path: str):
    state_dict = load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return models[arch].from_state_dict(state_dict).eval()

class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = [os.path.join(self.data_dir,f) for f in os.listdir(self.data_dir)]

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose(
        [transforms.CenterCrop(256), transforms.ToTensor()]
    )
        return transform(image)

    def __len__(self):
        return len(self.image_path)


def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("-m","--model",default="3anchorsbis",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-mp","--model_path",default="/scratch/inference/new_models/devil2022/",help="Model architecture (default: %(default)s)",)
    
    
    parser.add_argument("--lmbda", nargs='+', type=float, default =[0.025])
    
    parser.add_argument("-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",)

    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    parser.add_argument("-ni","--num_images",default = 48064, type = int)
    parser.add_argument("-niv","--num_images_val",default = 1024, type = int)

    parser.add_argument("-sp","--stanh_path",default="/scratch/inference/new_models/devil2022/3_anchors_stanh",help="Model architecture (default: %(default)s)",)#dddd
    parser.add_argument("-rp","--result_path",default="/scratch/inference/results",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-ip","--image_path",default="/scratch/dataset/kodak",help="Model architecture (default: %(default)s)",)
    parser.add_argument("--entropy_estimation", action="store_true", help="Use cuda")
    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--pretrained_stanh", action="store_true", help="Use cuda")
    parser.add_argument("--only_dist", action="store_true", help="Use cuda")
    parser.add_argument("--unfreeze_fact", action="store_true", help="Use cuda")
    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--filename",default="/data/",type=str,help="factorized_annealing",)
    
    parser.add_argument("-e","--epochs",default=600,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument("--fact_gp",default=15,type=int,help="factorized_beta",)
    parser.add_argument("--gauss_gp",default=15,type=int,help="gauss_beta",)

    parser.add_argument("--fact_annealing",default="gap_stoc",type=str,help="factorized_annealing",)
    parser.add_argument("--gauss_annealing",default="gap_stoc",type=str,help="factorized_annealing",)
    
    parser.add_argument("--num_stanh", type=int, default=1, help="Batch size (default: %(default)s)")
    parser.add_argument("--training_focus",default="stanh_levels",type=str,help="factorized_annealing",)

    args = parser.parse_args(argv) ###s
    return args


def bpp_calculation(out_net, out_enc):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]

        bpp_1 = (len(out_enc[0]) * 8.0 ) / num_pixels
        #print("la lunghezza è: ",len(out_enc[1]))
        bpp_2 =  sum( (len(out_enc[1][i]) * 8.0 ) / num_pixels for i in range(len(out_enc[1])))
        return bpp_1 + bpp_2, bpp_1, bpp_2


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics( org, rec, max_val: int = 255):
    metrics =  {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics


def read_image(filepath):
    #assert filepath.is_file()
    img = Image.open(filepath)
    img = img.convert("RGB")
    return transforms.ToTensor()(img)



def test_epoch(epoch, test_dataloader, model, criterion, valid,lmbda_list = None):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()


    psnr = AverageMeter()
    ssim = AverageMeter()
    with torch.no_grad():

        for i in range(model.num_stanh):
            print("valid level: ",i)
            for d in test_dataloader:
                d = d.to(device)

                out_net = model(d, training = False, stanh_level = i)
                out_criterion = criterion(out_net, d) if lmbda_list is None \
                                                        else criterion(out_net,d,lmbda = lmbda_list[i])
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])


                psnr.update(compute_psnr(d, out_net["x_hat"]))
                ssim.update(compute_msssim(d, out_net["x_hat"]))



    if valid is False:
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
        )
        log_dict = {
        "test":epoch,
        "test/loss": loss.avg,
        "test/bpp":bpp_loss.avg,
        "test/mse": mse_loss.avg,
        "test/psnr":psnr.avg,
        "test/ssim":ssim.avg,
        }
    else:

        print(
            f"valid epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
        )
        log_dict = {
        "valid":epoch,
        "valid/loss": loss.avg,
        "valid/bpp":bpp_loss.avg,
        "valid/mse": mse_loss.avg,
        "valid/psnr":psnr.avg,
        "valid/ssim":ssim.avg,
        }       

    wandb.log(log_dict)

    return loss.avg


def evaluation(model,filelist,entropy_estimation,device,epoch = -10):



    levels = [i for i in range(model.num_stanh)]

    psnr = [AverageMeter() for _ in range(model.num_stanh)]
    ms_ssim = [AverageMeter() for _ in range(model.num_stanh)]
    bpps =[AverageMeter() for _ in range(model.num_stanh)]



    bpp_across, psnr_across = [],[]
    for j in levels:
        print("***************************** ",j," ***********************************")
        for i,d in enumerate(filelist):
            name = "image_" + str(i)
            print(name," ",d," ",i)

            x = read_image(d).to(device)
            x = x.unsqueeze(0) 
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
            x_padded = F.pad(x, pad, mode="constant", value=0)

            

            if entropy_estimation is False: #ddd
                #print("entro qua!!!!")
                data =  model.compress(x_padded)
                out_dec = model.decompress(data)

            else:
                with torch.no_grad():
                    #print("try to do ",d)
                    out_dec = model(x_padded, training = False, stanh_level = j)
                    #print("done, ",d)
            if entropy_estimation is False:
                out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
                out_dec["x_hat"].clamp_(0.,1.)
                metrics = compute_metrics(x, out_dec["x_hat"], 255)
                size = out_dec['x_hat'].size()
                num_pixels = size[0] * size[2] * size[3]

                bpp ,_, _= bpp_calculation(out_dec, data["strings"]) #ddd

                
                metrics = compute_metrics(x_padded, out_dec["x_hat"], 255)
                print("fine immagine: ",bpp," ",metrics)

            else:
                out_dec["x_hat"].clamp_(0.,1.)
                out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
                size = out_dec['x_hat'].size()
                num_pixels = size[0] * size[2] * size[3]
                bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_dec["likelihoods"].values())
                metrics = compute_metrics(x, out_dec["x_hat"], 255)
                #print("fine immagine: ",bpp," ",metrics)
            

            
            bpps[j].update(bpp)
            psnr[j].update(metrics["psnr"]) #fff

            clear_memory()
        
            if epoch > -1:
                log_dict = {
                "compress":epoch,
                "compress/bpp_stanh_" + str(j): bpp,
                "compress/PSNR_stanh_" + str(j): metrics["psnr"],
                #"train/beta": annealing_strategy_gaussian.beta
                }

                wandb.log(log_dict)



        bpp_across.append(bpps[j].avg)
        psnr_across.append(psnr[j].avg)


        
    

    print(bpp_across," RISULTATI ",psnr_across)
    return bpp_across, psnr_across

def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])


def clear_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()




def main(argv):
    set_seed()
    args = parse_args(argv)

    if isinstance(args.lmbda,list):
        wandb.init(project="StanH_MultipleStairs",config = args, entity="albipresta") 
    else: 
        wandb.init(project="StanH_DecoderFineTuning",config = args, entity="albipresta")  


    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )


    model_name = args.model  # nome del modello che voglio studiare (ad esempio cheng2020)
    models_path = join(args.model_path,model_name) # percorso completo per arrivare ai modelli salvati (/scratch/inference/pretrained_models/chegn2020) qua ho salvato i modelli 
    device = "cuda"

    model_checkpoint = models_path + "/derivations/q4-a2-zou22.pth.tar"#a1-zou22.pth.tar" # this is the 
    checkpoint = torch.load(model_checkpoint, map_location=device)

    checkpoint["state_dict"]["gaussian_conditional._cdf_length"] = checkpoint["state_dict"]["gaussian_conditional._cdf_length"].ravel() #ffff



    factorized_configuration =checkpoint["factorized_configuration"]    
    gaussian_configuration =  checkpoint["gaussian_configuration"]#sssssdddd



    print("gaussian configuration: ",gaussian_configuration)


    factorized_configuration["beta"] = 10
    factorized_configuration["trainable"] = True
    factorized_configuration["annealing"] = args.fact_annealing
    factorized_configuration["gap_factor"] = args.fact_gp


                
    gaussian_configuration["beta"] = 10
    gaussian_configuration["trainable"] = True
    gaussian_configuration["annealing"] = args.gauss_annealing
    gaussian_configuration["gap_factor"] = args.gauss_gp





    if args.pretrained_stanh:
        print("entro qua!!!!")
        stanh_checkpoints_p = [args.stanh_path + "/anchors/a2-stanh.pth.tar",args.stanh_path + "/derivations/q4-a2-stanh.pth.tar",
                         args.stanh_path + "/derivations/q3-a2-stanh.pth.tar"]
    


        
        stanh_checkpoints = []

        for p in stanh_checkpoints_p:
            stanh_checkpoints.append(torch.load(p, map_location=device)) #ddd

    else:
        print("dovrebbe essere corretto, perché")
        stanh_checkpoints_p = args.stanh_path + "/anchors/a2-stanh.pth.tar"#a3-stanh.pth.tar" #q6-stanh.pth.tar"#a1-stanh.pth.tar"
        stanh_checkpoints = []

        for _ in range(args.num_stanh):
            stanh_checkpoints.append(torch.load(stanh_checkpoints_p, map_location=device))



    #define model 
    architecture =  models["cnn_multi"]

    factorized_configurations, gaussian_configurations = [],[]
    for jj in range(args.num_stanh) :

        factorized_configurations.append(factorized_configuration)
        gaussian_configurations.append(gaussian_configuration)
        print("DONE")

    
    model =architecture(N = 192, 
                            M = 320, 
                            num_stanh = args.num_stanh,
                            factorized_configuration = factorized_configurations, 
                            gaussian_configuration = gaussian_configurations #dddd
                           )
            

    model = model.to(device)
    model.update()

    ########### LOADING EVERYTHING ELSE!
    checkpoint["state_dict"] = update_checkpopoint(checkpoint["state_dict"],num_stanh = args.num_stanh)
    model.load_state_dict(checkpoint["state_dict"],stanh_checkpoints)


    images_path = args.image_path # path del test set 
    #savepath = args.result_path # path dove salvare i risultati 
    image_list = [os.path.join(images_path,f) for f in listdir(images_path)]
    
    model.freeze_net()
    bpp_init, psnr_init = evaluation(model,image_list,entropy_estimation = args.entropy_estimation,device = device, epoch = -10)

    print("finita valutazione iniziale: ",bpp_init," ",psnr_init) #sss

    if args.training_focus == "tune_gs":
        model.unfreeze_decoder()
    elif args.training_focus == "stanh_levels":
        model.unfreeze_quantizer(args.unfreeze_fact)
    elif args.training_focus == "all":
        model.unfreeze_all()
    else:
        model.unfreeze_quantizer()
        model.unfreeze_decoder()

    if isinstance(args.lmbda,list):
    
        criterion = RateDistortionLoss(lmbda=args.lmbda)
    else:
        criterion = RateDistortionLoss(lmbda=args.lmbda,only_dist = args.only_dist)

    optimizer = configure_optimizers(model, args)
    annealing_z, annealing_y = [],[]

    for ii in range(model.num_stanh):
        annealing_strategy_bottleneck, annealing_strategy_gaussian =  configure_annealings(factorized_configuration, gaussian_configuration)
        annealing_z.append(annealing_strategy_bottleneck)
        annealing_y.append(annealing_strategy_gaussian)




    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms, num_images=args.num_images)
    valid_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms, num_images=args.num_images_val)
    test_dataset = TestKodakDataset(data_dir="/scratch/dataset/kodak")
    device = "cuda" 

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,

    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )


    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=10)




    previous_lr = optimizer.param_groups[0]['lr']
    print("subito i paramteri dovrebbero essere giusti!")
    model_tr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_fr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad== False)
    #fact_gp = args.fact_gp
    #gauss_gp = args.gauss_gp

    print("trainable pars: ",model_tr_parameters)
    print("frozen pars: ",model_fr_parameters)



    last_epoch = 0
    counter = 0
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print("**************** epoch: ",epoch,". Counter: ",counter)
        previous_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}","    ",previous_lr)
        print("epoch ",epoch)
        start = time.time()
        counter = train_one_epoch(
            model,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            counter,
            annealing_z, 
            annealing_y,
            lmbda_list=args.lmbda

        )

        loss_valid = test_epoch(epoch, valid_dataloader, model, criterion,  valid = True, lmbda_list = args.lmbda)

        loss = test_epoch(epoch, test_dataloader, model, criterion,  valid = False,  lmbda_list = args.lmbda)

        lr_scheduler.step(loss_valid)

        print(" execute  the test to verify")

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        filename, filename_best =  create_savepath(args)


        if (is_best) or epoch%5==0:
            save_checkpoint_our(
                            {
                                "epoch": epoch,
                                "annealing_strategy_bottleneck":annealing_strategy_bottleneck,
                                "annealing_strategy_gaussian":annealing_strategy_gaussian,
                                "state_dict": model.state_dict(),
                                "loss": loss,
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "factorized_configuration": model.factorized_configuration,
                                "gaussian_configuration":model.gaussian_configuration,
                                "entropy_bottleneck_w":[model.entropy_bottleneck[i].sos.w for i in range(model.num_stanh)],
                                "entropy_bottleneck_b":[model.entropy_bottleneck[i].sos.b for i in range(model.num_stanh)],
                                "gaussian_conditional_w":[model.gaussian_conditional[i].sos.w for i in range(model.num_stanh)],
                                "gaussian_conditional_b":[model.gaussian_conditional[i].sos.b for i in range(model.num_stanh)],

                        },
                        is_best,
                        filename,
                        filename_best    
                    )  


        psnr_res = {}
        bpp_res = {}

        model.update()
        bpp_post, psnr_post = evaluation(model,image_list,entropy_estimation = args.entropy_estimation,device = device, epoch = epoch)
        bpp_res["our_init"] = bpp_init
        psnr_res["our_init"] = psnr_init

        bpp_res["our_post"] = bpp_post
        psnr_res["our_post"] = psnr_post


        psnr_res["base"] =   [32.26,34.15]
        bpp_res["base"] =  [0.309,0.449]
        print("log also the current leraning rate")

        plot_rate_distorsion(bpp_res, psnr_res,epoch, eest="compression")
        #plot_sos(model, device)
        endt = time.time()

        print("Runtime of the epoch  ", epoch)
        sec_to_hours(endt - start) 

        log_dict = {
        "train":epoch,
        "train/leaning_rate": optimizer.param_groups[0]['lr'],
        #"train/beta": annealing_strategy_gaussian.beta
        }

        wandb.log(log_dict)

if __name__ == "__main__":

     
    main(sys.argv[1:])