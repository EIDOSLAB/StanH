
import torch 
import os 
import numpy as np 

from compress.utils.help_function import compute_msssim, compute_psnr
from torchvision import transforms
from PIL import Image
import torch


from compress.models.cnn_multiStanh import WACNNMultiSos
import torch.nn.functional as F
import math
from compressai.ops import compute_padding
import math 
from pytorch_msssim import ms_ssim
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
from compressai.zoo import *
from os.path import join 
import wandb
from os import listdir
from collections import OrderedDict
from compress.utils.annealings import *
from compress.zoo import *
torch.backends.cudnn.benchmark = True #sss



import torch.nn as nn
from pytorch_msssim import ms_ssim 


from os.path import join 
import wandb
import seaborn as sns
palette = sns.color_palette("tab10")
import matplotlib.pyplot as plt






import numpy as np
import scipy.interpolate


def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)

    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), np.sort(PSNR1), samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), np.sort(PSNR2), samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_diff = (int2-int1)/(max_int-min_int)

    return avg_diff


def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), np.sort(lR1), samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), np.sort(lR2), samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2-int1)/(max_int-min_int)
    avg_diff = (np.exp(avg_exp_diff)-1)*100
    return avg_diff




def plot_rate_distorsion(bpp_res, psnr_res,epoch, eest = "compression", index_list = [4]):

    chiavi_da_mettere = list(psnr_res.keys())
    legenda = {}
    for i,c in enumerate(chiavi_da_mettere):
        legenda[c] = {}
        legenda[c]["colore"] = [palette[i],'-']
        legenda[c]["legends"] = c
        legenda[c]["symbols"] = ["*"]*300
        legenda[c]["markersize"] = [5]*300    

    plt.figure(figsize=(12,8)) # fig, axes = plt.subplots(1, 1, figsize=(8, 5))#dddd

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
        plt.plot(bpp,psnr,"-" ,color = colore, label =  leg ,markersize=10)       
        plt.plot(bpp, psnr, marker="s", markersize=4, color =  colore)


        if "proposed" in type_name:
            for jjj in index_list:
                plt.plot(bpp[jjj], psnr[jjj], marker="*", markersize=12, color =  colore)
                plt.plot(bpp[jjj], psnr[jjj], marker="*", markersize=12, color =  colore)
                plt.plot(bpp[jjj], psnr[jjj], marker="*", markersize=12, color =  colore) #fff

            #for jjj in [9,18]:
            #    plt.plot(bpp[jjj], psnr[jjj], marker="o", markersize=6, color =  colore)
            #    plt.plot(bpp[jjj], psnr[jjj], marker="o", markersize=6, color =  colore)
            #    plt.plot(bpp[jjj], psnr[jjj], marker="o", markersize=6, color =  colore) #fff



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
    
    
    parser.add_argument("--lmbda", nargs='+', type=float, default =[0.0009, 0.0035,0.0067,0.025, 0.0820,0.15])
    
    parser.add_argument("-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",)

    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    parser.add_argument("-ni","--num_images",default = 48064, type = int)
    parser.add_argument("-niv","--num_images_val",default = 1024, type = int)

    parser.add_argument("-sp","--stanh_path",default="/scratch/inference/new_models/devil2022/3_anchors_stanh",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-rp","--result_path",default="/scratch/inference/results",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-ip","--image_path",default="/scratch/dataset/kodak",help="Model architecture (default: %(default)s)",)
    parser.add_argument("--entropy_estimation", action="store_true", help="Use cuda")
    parser.add_argument("--pretrained_stanh", action="store_true", help="Use cuda")
    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)

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
    
    parser.add_argument("--num_stanh", type=int, default=7, help="Batch size (default: %(default)s)")
    parser.add_argument("--training_focus",default="stanh_levels",type=str,help="factorized_annealing",)

    args = parser.parse_args(argv) ###s
    return args





def update_checkpopoint(state_dict,num_stanh):

    res =  OrderedDict()


    for k,v in state_dict.items():
        if "gaussian_conditional" in k:
            continue
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


def evaluation(model,filelist,entropy_estimation,device,epoch = -10, custom_levels = None):



    levels = [i for i in range(model.num_stanh)] if custom_levels is None else custom_levels

    psnr = [AverageMeter() for _ in range(model.num_stanh)] if custom_levels is None else [AverageMeter() for _ in range(len(custom_levels))]
    
    bpps =[AverageMeter() for _ in range(model.num_stanh)] if custom_levels is None else [AverageMeter() for _ in range(len(custom_levels))]



    bpp_across, psnr_across = [],[]
    cont = 0
    for j in levels:
        print("***************************** ",j," ***********************************") #fff
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
                metrics = compute_metrics(x, out_dec["x_hat"], 255)#ddddd
                #print("fine immagine: ",bpp," ",metrics)
            

            
            bpps[cont].update(bpp.item())
            psnr[cont].update(metrics["psnr"]) #fff


        
            if epoch > -1:
                log_dict = {
                "compress":epoch,
                "compress/bpp_stanh_" + str(cont): bpp,
                "compress/PSNR_stanh_" + str(cont): metrics["psnr"],
                #"train/beta": annealing_strategy_gaussian.beta
                }

                wandb.log(log_dict)



        bpp_across.append(bpps[cont].avg)
        psnr_across.append(psnr[cont].avg)

        cont = cont +1


        
    

    print(bpp_across," RISULTATI ",psnr_across)
    return bpp_across, psnr_across



def main(argv):
    set_seed()
    args = parse_args(argv)


    wandb.init(project="eval_multi",config = args, entity="albipresta") 


    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )


    model_name = args.model  # nome del modello che voglio studiare (ad esempio cheng2020)
    models_path = join(args.model_path,model_name) # percorso completo per arrivare ai modelli salvati (/scratch/inference/pretrained_models/chegn2020) qua ho salvato i modelli 
    device = "cuda"

    model_checkpoint = models_path + "/anchors/q5-zou22.pth.tar" # this is the 
    checkpoint = torch.load(model_checkpoint, map_location=device)

    checkpoint["state_dict"]["gaussian_conditional._cdf_length"] = checkpoint["state_dict"]["gaussian_conditional._cdf_length"].ravel()
    factorized_configuration =checkpoint["factorized_configuration"]
    factorized_configuration["trainable"] = True
    gaussian_configuration =  checkpoint["gaussian_configuration"]#sssss
    gaussian_configuration["trainable"] = True #ddd


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
            stanh_checkpoints.append(torch.load(p, map_location=device))
    else:
        stanh_checkpoints_p = args.stanh_path + "/anchors/a2-stanh.pth.tar"
        stanh_checkpoints = []

        


        #cc = torch.load(stanh_checkpoints_p)
        #print("cc---------------------- ",cc["gaussian_configuration"])
        factorized_configuration = []
        gaussian_configuration = []
        for jj in [1,1.5,0,2.5,3,3.5,4]:#range(args.num_stanh):
            d = torch.load(stanh_checkpoints_p, map_location=device) 
            d["gaussian_configuration"]["num_sigmoids"] = int(d["gaussian_configuration"]["extrema"]*jj)
            print("--------------------------------")
            print(d["gaussian_configuration"])
            stanh_checkpoints.append(d)

            factorized_configuration.append(d["factorized_configuration"])
            gaussian_configuration.append(d["gaussian_configuration"])
            print("DONE")


   
    model =WACNNMultiSos(N = 192, 
                            M = 320, 
                            num_stanh = args.num_stanh,
                            factorized_configuration = factorized_configuration, 
                            gaussian_configuration = gaussian_configuration)#ddd
            

    model = model.to(device)
    model.update()

    ########### LOADING EVERYTHING ELSE!
    checkpoint["state_dict"] = update_checkpopoint(checkpoint["state_dict"],num_stanh = args.num_stanh)
    model.load_state_dict(checkpoint["state_dict"],state_dicts_stanh = None)#stanh_checkpoints



    images_path = args.image_path # path del test set 
    #savepath = args.result_path # path dove salvare i risultati 
    image_list = [os.path.join(images_path,f) for f in listdir(images_path)]
    
    model.freeze_net()



    adding_levels = [0.0005,0.00075,0.001,0.0011,0.00115,0.0012,0.00125,0.0014]
    start_levels = [0,1,2,3,4,5,6]
    custom_levels = []#[0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.1,1,1.001,1.1,2]


    for i in start_levels:
        custom_levels.append(i)
        for j in adding_levels:
            custom_levels.append(i + j)
    
    #custom_levels.append(2)
    

    print(custom_levels)
    bpp_, psnr_ = evaluation(model,image_list,entropy_estimation = args.entropy_estimation,device = device, epoch = -10, custom_levels=custom_levels)
    print("DONE") #dddd


    psnr_res = {}
    bpp_res = {}

    bpp_res["gain"] = [0.23839285714285716,  0.3410714285714286, 0.47410714285714284]
    psnr_res["gain"] = [ 30.805755395683455,  32.34532374100719, 33.94244604316547]

    bpp_res["manual"] = bpp_
    psnr_res["manual"] = psnr_




    bpp_res["proposed"] = [0.2432186234817814, 0.2595141700404858, 0.29281376518218627, 0.33461538461538465, 0.4012145748987854, 0.4706477732793522, 0.5918016194331983, 0.6506072874493927, 0.7802631578947368]
    psnr_res["proposed"] = [30.70534351145038, 31.33587786259542, 31.91297709923664, 32.62900763358779, 33.56946564885496, 34.210687022900764, 34.894656488549614, 35.14045801526718, 35.4824427480916]

    plot_rate_distorsion(bpp_res, psnr_res,0, eest="compression")


    print("Our-adapt")
    print('BD-PSNR: ', BD_PSNR(bpp_res["manual"], psnr_res["manual"], bpp_res["proposed"],psnr_res["proposed"]))
    print('BD-RATE: ', BD_RATE(bpp_res["manual"], psnr_res["manual"], bpp_res["proposed"], psnr_res["proposed"]))




if __name__ == "__main__":
    main(sys.argv[1:])