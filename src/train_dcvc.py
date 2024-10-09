import torch 
import os 
import numpy as np 
import random
from compress.utils.help_function import compute_msssim, compute_psnr
from torchvision import transforms
from PIL import Image
import torch
import time
from compress.datasets import ImageFolder, TestKodakDataset
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
import wandb
from os import listdir
from compress.utils.annealings import *
from compress.models.DCVC.dcvc import IntraNoAR
from compress.models.DCVC.wacnn_dcvc import WACNN_DCVC



torch.backends.cudnn.benchmark = True #sss



import torch.nn as nn
from pytorch_msssim import ms_ssim 

from datetime import datetime
from os.path import join 
import wandb
import shutil







def save_checkpoint(state, is_best, filename,filename_best):
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

    def __init__(self, lmbda = 1e-2,  metric = "mse" ): #ddd
        super().__init__()


        if metric is "mse":
            self.dist_metric = nn.MSELoss()
        else:
            self.dist_metric = ms_ssim 

        self.lmbda = lmbda[0] if isinstance(lmbda,list) else lmbda




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
    if args.model == "dcvc":
        parameters = {
            n
            for n, p in net.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        }


        pars = {
            n
            for n, p in net.named_parameters()
            if  n.endswith(".quantiles")
        }

        print("lunghezza dict: ",len(pars))

        params_dict = dict(net.named_parameters())
        optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)),lr=args.learning_rate,)
        return optimizer,None 
    else:
        parameters = {
            n
            for n, p in net.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        }
        aux_parameters = {
            n
            for n, p in net.named_parameters()
            if n.endswith(".quantiles") and p.requires_grad
        }

        # Make sure we don't have an intersection of parameters
        params_dict = dict(net.named_parameters())
        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters

        assert len(inter_params) == 0
        assert len(union_params) - len(params_dict.keys()) == 0

        optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)),lr=args.learning_rate,)
        aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)),lr=args.aux_learning_rate,)

        return optimizer, aux_optimizer







import seaborn as sns
palette = sns.color_palette("tab10")
import matplotlib.pyplot as plt








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



def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)




def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")


    
    parser.add_argument("--lmbda_list", nargs='+', type=float, default =[0.0025, 0.0067, 0.025,0.05])
    parser.add_argument("--q_global", nargs='+', type=float, default =[1.5409, 1.0826, 0.7293, 0.5000])
    
    parser.add_argument("-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",)

    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    parser.add_argument("-ni","--num_images",default = 300000, type = int)
    parser.add_argument("-niv","--num_images_val",default = 1024, type = int)
    parser.add_argument("--N",default = 192, type = int)
    parser.add_argument("--M",default = 320, type = int)
    parser.add_argument("--anchor_num",default = 4, type = int)

    parser.add_argument( "--aux-learning-rate", default=1e-3, type=float, help="Auxiliary loss learning rate (default: %(default)s)",)

    parser.add_argument( "--model", type=str, default = "dcvc_wacnn", help="Training dataset")#ddddd




    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--filename",default="/scratch/DCVC/HEM/files",type=str,help="factorized_annealing",)
    
    parser.add_argument("-e","--epochs",default=600,type=int,help="Number of epochs (default: %(default)s)",)

    parser.add_argument("--rate_num",default=8,type=int,help="rate_num",)


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


def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])




def train_one_epoch(model, 
                    criterion, 
                    train_dataloader,
                    optimizer,
                    aux_optimizer,
                    epoch, 
                    clip_max_norm,
                    counter, 
                    lmbda_list ):
    model.train()
    device = next(model.parameters()).device

  
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()



    for i, d in enumerate(train_dataloader):
        counter += 1
        d = d.to(device)

        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()

        quality_index =  random.randint(0, len(model.lmbda_list) - 1)
        #quality_index =  1#random.randint(0, model.num_stanh - 1)
        out_net = model(d, index = quality_index)


        out_criterion = criterion(out_net, d, lmbda_list[quality_index]) 
        out_criterion["loss"].backward()

        loss.update(out_criterion["loss"].clone().detach())
        mse_loss.update(out_criterion["mse_loss"].clone().detach())
        bpp_loss.update(out_criterion["bpp_loss"].clone().detach())


        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()


        if aux_optimizer is not None:
            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()



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



        

    log_dict = {
        "train":epoch,
        "train/loss": loss.avg,
        "train/bpp": bpp_loss.avg,
        "train/mse": mse_loss.avg,
        }
        
    wandb.log(log_dict)
    return counter


def test_epoch(epoch, test_dataloader, model, criterion,lmbda_list,valid):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()


    psnr = AverageMeter()
    ssim = AverageMeter()
    with torch.no_grad():

        for i,qs in enumerate(model.q_scale.ravel()):
            print("valid level: ",i)
            for d in test_dataloader:
                d = d.to(device)

                out_net = model(d,  q_scale =  qs.item())
                out_criterion = criterion(out_net,d,lmbda = lmbda_list[i])
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



def interpolate_log(min_val, max_val, num, decending=True):
    assert max_val > min_val
    assert min_val > 0
    if decending:
        values = np.linspace(np.log(max_val), np.log(min_val), num)
    else:
        values = np.linspace(np.log(min_val), np.log(max_val), num)
    values = np.exp(values)
    return values


def run_inference(model, image_list,rate_num,device):


    


    max_q_scale = model.q_scale.ravel()[0].item()
    min_q_scale = model.q_scale.ravel()[-1].item()
    i_frame_q_scales = interpolate_log(min_q_scale, max_q_scale, rate_num) #dddd

    psnrs = [AverageMeter() for _ in range(len(i_frame_q_scales))]
    bpps =   [AverageMeter() for _ in range(len(i_frame_q_scales))]
    for j,q in enumerate(i_frame_q_scales):

        for i,d in enumerate(image_list):
            x = read_image(d).to(device)
            x = x.unsqueeze(0)
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)
            x_padded = F.pad(x, pad, mode="constant", value=0) #dddd


            with torch.no_grad():
                data = model.compress(x_padded, q)
                result = model.decompress(data["strings"],data["shape"],q)
            
            
            result["x_hat"] = F.pad(result["x_hat"], unpad)
            metrics = compute_metrics(x, result["x_hat"], 255)


            size = result["x_hat"].size()
            num_pixels = size[0] * size[2] * size[3]
            bpp = sum(len(s[0]) for s in data["strings"]) * 8.0 / num_pixels




            psnrs[j].update(metrics["psnr"])
            bpps[j].update(bpp)

    
    bpp_avg = [bpps[i].avg for i in range(len(q_scales))]
    psnr_avg = [psnrs[i].avg for i in range(len(q_scales))]

    print("------ ",bpp_avg)
    print("----- ",psnr_avg)
    return bpp_avg, psnr_avg



import seaborn as sns
palette = sns.color_palette("tab10")
import matplotlib.pyplot as plt


def plot_rate_distorsion(bpp_res, psnr_res,epoch, eest = "compression"):

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






def main(argv):
    set_seed()
    device = "cuda"
    args = parse_args(argv)


    wandb.init(project="HEM-training",config = args, entity="albipresta") 





    train_transforms = transforms.Compose( [transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])



    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms, num_images=args.num_images)
    valid_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms, num_images=args.num_images_val)
    test_dataset = TestKodakDataset(data_dir="/scratch/dataset/kodak")



    image_list = [os.path.join("/scratch/dataset/kodak",f) for f in listdir("/scratch/dataset/kodak")]

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


   





    if args.model  == "dcvc":
        net = IntraNoAR(N = args.N, anchor_num = args.anchor_num, q_global = args.q_global, lmbda_list= args.lmbda_list)
        params = {"N":args.N, "anchor_num":args.anchor_num,"q_global":args.q_global, "lmbda_list":args.lmbda_list}
    else:
        net =  WACNN_DCVC(N = args.N,M = args.M,anchor_num = args.anchor_num, lmbda_list= args.lmbda_list)
        params = {"N":args.N, "anchor_num":args.anchor_num,"M":args.M, "lmbda_list":args.lmbda_list}
    
    net.to(device)
    net.update()


    criterion = RateDistortionLoss(lmbda=args.lmbda_list)
    optimizer,aux_optimizer = configure_optimizers(net, args)


    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=4)
    
    previous_lr = optimizer.param_groups[0]['lr']
    print("subito i paramteri dovrebbero essere giusti!")
    model_tr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    model_fr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad== False)



    last_epoch = 0
    counter = 0
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print("**************** epoch: ",epoch,". Counter: ",counter)
        
        print("trainable pars: ",model_tr_parameters)
        print("frozen pars: ",model_fr_parameters)


        previous_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}","    ",previous_lr)

        start = time.time()
        counter = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            1.0,
            counter,
            lmbda_list=args.lmbda_list

        )


        loss_valid = test_epoch(epoch, valid_dataloader, net, criterion,args.lmbda_list,  valid = True)
        lr_scheduler.step(loss_valid)
        loss = test_epoch(epoch, test_dataloader, net, criterion, args.lmbda_list, valid = False)


        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        filename, filename_best =  create_savepath(args)



        if (is_best) or epoch%5==0:
            save_checkpoint(
                            {
                                "epoch": epoch,
                                "input_pars":params,
                                "state_dict": net.state_dict(),
                                "loss": loss,
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),


                        },
                        is_best,
                        filename,
                        filename_best    
                    )  


        psnr_res = {}
        bpp_res = {}

        net.update() 


        bpp_, psnr_ = run_inference(net,image_list,args.rate_num,device)  


        bpp_res["DCVC"] = bpp_ 
        psnr_res["DCVC"] = psnr_

        bpp_res["proposed_1A"] = [0.2413, 0.26, 0.2912, 0.3328, 0.40, 0.4714, 0.5909, 0.6509, 0.778]  
        psnr_res["proposed_1A"] = [30.72, 31.34, 31.904, 32.630, 33.54, 34.214, 34.901, 35.17, 35.4565]


        bpp_res["proposed_3A"] = [0.11873141724479683, 0.1499504459861249,0.2413, 0.26, 0.2912, 0.3328, 0.40, 0.4714, 0.5909, 0.6509,0.7324715615305067, 0.7841778697001034,0.8403369672943508, 0.9144777662874871]
        psnr_res["proposed_3A"] = [28.663780663780663, 29.37085137085137,30.72, 31.34, 31.904, 32.630, 33.54, 34.214, 34.901, 35.14448, 36.345238095238095, 36.773809523809526, 37.23520923520923, 37.68452380952381]


        plot_rate_distorsion(bpp_res, psnr_res,epoch, eest="compression")


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