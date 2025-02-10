import torch
import time
from os.path import join 
from os import listdir
from torch.utils.data import DataLoader
from compress.datasets import ImageFolder
from compress.utils.help_function import  configure_latent_space_policy
import numpy as np
from compress.zoo import models, aux_net_models
from compressai.zoo import *
from torchvision.transforms import RandomCrop, Compose, ToTensor
from compress.utils.annealings import *
from compress.utils.stf.loop import train_one_epoch, configure_optimizers, test_epoch, evaluation
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



def main(argv):
    set_seed()
    args = parse_args(argv)
    device = args.device 


    wandb.init(project="StanH_MultipleStairs",config = args) 



    train_transforms = Compose( [RandomCrop(args.patch_size), ToTensor()])

    test_transforms = Compose([RandomCrop(args.patch_size), ToTensor()])


    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms, num_images=args.num_images)
    valid_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms, num_images=args.num_images_val)
    test_dataset = TestKodakDataset(data_dir=args.test_datapath)


    

    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,)

    valid_dataloader = DataLoader(valid_dataset,batch_size=args.batch_size,num_workers=args.num_workers, shuffle=False,)



    _ , gaussian_configuration = configure_latent_space_policy(args)
    
    gaussian_configurations = [gaussian_configuration for _ in range(args.num_stanh)]

    model = models["stf_StanH"](
                            num_stanh = args.num_stanh,
                            gaussian_configuration = gaussian_configurations #dddd
                           )
    
    ########### LOADING EVERYTHING ELSE!
    checkpoint = torch.load(args.anchor_path , map_location=device)
    checkpoint["state_dict"] = InsertStanHOnCheckpoints(checkpoint["state_dict"],num_stanh = args.num_stanh)
    model.load_state_dict(checkpoint["state_dict"], strict = False)
    
    
    images_path = args.image_path # path del test set 
    image_list = [join(images_path,f) for f in listdir(images_path)]  
    model.freeze_net()
    bpp_init, psnr_init = evaluation(model,image_list,entropy_estimation = args.entropy_estimation,device = device, epoch = -10)

    print("finita valutazione iniziale: ",bpp_init," ",psnr_init) #sss