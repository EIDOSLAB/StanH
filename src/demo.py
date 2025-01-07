
import torch 
import os 
import numpy as np 

from torchvision import transforms
from PIL import Image
import torch
import json
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
import wandb
from os import listdir
from compress.utils.annealings import *
from compress.models.cnn_multiStanh import WACNNMultiSTanH
torch.backends.cudnn.benchmark = True #sss
from pytorch_msssim import ms_ssim 
import wandb
import seaborn as sns
palette = sns.color_palette("tab10")
import matplotlib.pyplot as plt



import numpy as np



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

def evaluation(model,filelist,entropy_estimation,device,epoch = -10, custom_levels = None):



    levels = [i for i in range(model.num_stanh)] if custom_levels is None else custom_levels
    psnr = [AverageMeter() for _ in range(model.num_stanh)] if custom_levels is None else [AverageMeter() for _ in range(len(custom_levels))]
    bpps =[AverageMeter() for _ in range(model.num_stanh)] if custom_levels is None else [AverageMeter() for _ in range(len(custom_levels))]



    bpp_across, psnr_across = [],[]
    cont = 0
    for j in levels:
        for i,d in enumerate(filelist):
            name = "image_" + str(i)

            x = read_image(d).to(device)
            x = x.unsqueeze(0) 
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
            x_padded = F.pad(x, pad, mode="constant", value=0)

            

            if entropy_estimation is False: #ddd

                data =  model.compress(x_padded,stanh_level = j)
                out_dec = model.decompress(data,stanh_level = j)

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


            else:
                out_dec["x_hat"].clamp_(0.,1.)
                out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
                size = out_dec['x_hat'].size()
                num_pixels = size[0] * size[2] * size[3]
                bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_dec["likelihoods"].values())
                bpp = bpp.item()
                metrics = compute_metrics(x, out_dec["x_hat"], 255)#ddddd
                #print("fine immagine: ",bpp," ",metrics)
            

            
            bpps[cont].update(bpp)
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


        
    

    #print(bpp_across," RISULTATI ",psnr_across)
    return bpp_across, psnr_across




def plot_rate_distorsion(bpp_res, psnr_res,epoch, eest = "compression", index_list = []):

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
        plt.plot(bpp,psnr,"-" ,color = colore, label =  leg ,markersize=4)       
        plt.plot(bpp, psnr, marker="o",markerfacecolor='none' if "proposed" in type_name else colore , markersize=4, color =  colore) 


        if "proposed" in type_name:
            for jjj in index_list:
                plt.plot(bpp[jjj], psnr[jjj], marker="o", markersize=8, color =  colore)
                plt.plot(bpp[jjj], psnr[jjj], marker="o", markersize=8, color =  colore)
                plt.plot(bpp[jjj], psnr[jjj], marker="o", markersize=8, color =  colore) #fff



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
    psnr_tick =  [round(x) for x in range(minimo_psnr, massimo_psnr + 1)]
    plt.ylabel('PSNR', fontsize = 30)
    plt.yticks(psnr_tick)

    #print(minimo_bpp,"  ",massimo_bpp)

    bpp_tick =   [round(x)/10 for x in range(int(minimo_bpp*10), int(massimo_bpp*10 + 1))]
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


def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    
    parser.add_argument("-mc","--model_checkpoint",default="/scratch/inference/new_models/devil2022/rebuttal_model/A40/model/0728_last_.pth.tar",help="path to test mode",)
    parser.add_argument("-sp","--stanh_path",default="/scratch/inference/new_models/devil2022/rebuttal_model/A40/stanh",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-rp","--save_path",default=None,help="wehre to save results",)
    parser.add_argument("-ip","--image_path",default="/scratch/dataset/kodak",help="Model architecture (default: %(default)s)",)
    parser.add_argument("--device",default="cuda",help="device (cuda or cpu)",)
    parser.add_argument("--entropy_estimation", action="store_true", help="Use cuda")
    parser.add_argument("--wandb_log", action="store_true", help="Use cuda")

    args = parser.parse_args(argv) ###s
    return args


def main(argv):
    set_seed()
    args = parse_args(argv)

    if args.wandb_log:
        wandb.init(project="TEST-STANH",config = args, entity="alberto-presta") 




    device = args.device
    stanh_cheks = [os.path.join(args.stanh_path,f) for f in sorted(os.listdir(args.stanh_path))]
    model_checkpoint = args.model_checkpoint #"/scratch/inference/new_models/devil2022/rebuttal_model/A40/model/0728_last_.pth.tar"   #ffff
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model =WACNNMultiSTanH(N = 192, 
                            M = 320, 
                            num_stanh = len(stanh_cheks),
                            factorized_configuration = checkpoint["factorized_configuration"], 
                            gaussian_configuration = checkpoint["gaussian_configuration"])#ddd
            

    model = model.to(device)
    model.update()



    model.load_state_dict(checkpoint["state_dict"],state_dicts_stanh = None)

    


    for i in range(len(stanh_cheks)):
     
        print(stanh_cheks[i])
        sc = stanh_cheks[i]#os.path.join(stanh_path,stanh_cheks[i])
        

        stanhs = torch.load(sc,map_location=device)
        weights = stanhs["state_dict"]["gaussian_conditional"]["w"]
        biases = stanhs["state_dict"]["gaussian_conditional"]["b"]

    

        model.gaussian_conditional[i].sos.w = torch.nn.Parameter(weights)
        model.gaussian_conditional[i].sos.b = torch.nn.Parameter(biases)
        model.gaussian_conditional[i].sos.update_state()

        weights = stanhs["state_dict"]["entropy_bottleneck"]["w"]
        biases = stanhs["state_dict"]["entropy_bottleneck"]["b"]

        model.entropy_bottleneck[i].sos.w = torch.nn.Parameter(weights)
        model.entropy_bottleneck[i].sos.b = torch.nn.Parameter(biases)
        model.entropy_bottleneck[i].sos.update_state()
    
    model.update()

    

    adding_levels =  [] #None #[0.0001,0.0009,0.001,0.0012,0.00125,0.0014]#[0.0001,0.00025,0.0005,0.00075,0.00075,0.0009,0.001,0.0011,0.00115,0.0012,0.00125,0.0014]
    start_levels = [0,1,2,3,4,5,6]
    custom_levels = []


    for i in start_levels:
        custom_levels.append(i)
        for j in adding_levels:
            if i != 6:
                custom_levels.append(i + j)



    images_path = args.image_path
    image_list = [[os.path.join(images_path,f) for f in listdir(images_path)][0]] #ddd
    print(image_list)
    bpp_, psnr_ = evaluation(model,image_list,entropy_estimation = args.entropy_estimation,device = device, epoch = -10, custom_levels=custom_levels)
    
    
    
    print("DONE---> ",bpp_,"----",psnr_) #dddd
    if args.save_path is not None: 
        data = {
            "bpp": bpp_,
            "psr": psnr
        }


        file = os.path.join(args.save_path,"output.json")
        with open("output.json", "w") as file:
            json.dump(data, file, indent=4)

    psnr_res = {}
    bpp_res = {}




    indexes = []
    for i in start_levels:
        indexes.append(custom_levels.index(i))


    bpp_res["proposed"] = bpp_ 
    psnr_res["proposed"] =  psnr_ 



    if args.wandb_log:
        plot_rate_distorsion(bpp_res, psnr_res,0, eest="compression",index_list=[])



if __name__ == "__main__":
    main(sys.argv[1:])