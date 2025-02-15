import torch 
from PIL import Image 
import math
from pytorch_msssim import ms_ssim
from torchvision import transforms
from compress.utils.annealings import RandomAnnealings, Annealing_triangle, Annealings
import numpy as np
import wandb
import shutil
from datetime import datetime
from os.path import join 


def create_savepath(args):
    now = datetime.now()
    date_time = now.strftime("%m%d")
    suffix = ".pth.tar"
    c = join(date_time,"last").replace("/","_")

    
    c_best = join(date_time,"best").replace("/","_")
    c = join(c,suffix).replace("/","_" + args.lmbda[0])
    c_best = join(c_best,suffix).replace("/","_"+ args.lmbda[0])
    
    
    path = args.filename
    savepath = join(path,c)
    savepath_best = join(path,c_best)
    
    print("savepath: ",savepath)
    print("savepath best: ",savepath_best)
    return savepath, savepath_best

def configure_annealings( gaussian_configuration):

    if gaussian_configuration is None:
        annealing_strategy_gaussian = None 
    elif "random" in gaussian_configuration["annealing"]:
        annealing_strategy_gaussian = RandomAnnealings(beta = gaussian_configuration["beta"],  type = gaussian_configuration["annealing"], gap = False)
    elif "none" in gaussian_configuration["annealing"]:
        annealing_strategy_gaussian = None
    
    elif "triangle" in gaussian_configuration["annealing"]:
        annealing_strategy_gaussian = Annealing_triangle(beta = gaussian_configuration["beta"], 
                                                         factor = gaussian_configuration["gap_factor"])
    
    else:
        annealing_strategy_gaussian = Annealings(beta = gaussian_configuration["beta"], 
                                    factor = gaussian_configuration["gap_factor"], 
                                    type = gaussian_configuration["annealing"]) 
    

    return annealing_strategy_gaussian


def save_checkpoint_our(state, is_best, filename,filename_best):
    torch.save(state, filename)
    wandb.save(filename)
    if is_best:
        shutil.copyfile(filename, filename_best)
        wandb.save(filename_best)

def configure_latent_space_policy(args):
    
    gaussian_configuration = {
                "beta": 10, 
                "num_sigmoids": args.gauss_num_sigmoids, 
                "activation": args.gauss_activation, 
                "annealing": args.gauss_annealing, 
                "gap_factor": args.gauss_gp ,
                "extrema": args.gauss_extrema ,
                "trainable": True
     
            }


    return gaussian_configuration


def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)




def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])


def clear_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()





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
