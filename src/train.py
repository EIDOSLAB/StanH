# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import random
import shutil
import sys
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from compress.training.step import train_one_epoch, test_epoch, compress_with_ac
from compress.training.loss import RateDistortionLoss
from torch.utils.data import DataLoader
from torchvision import transforms
import os 
import glob
from compress.datasets import ImageFolder
from compress.zoo import models, aux_net_models
from compress.utils.annealings import *
from compress.utils.help_function import CustomDataParallel, configure_optimizers, save_checkpoint, configure_latent_space_policy, create_savepath, save_checkpoint_our, sec_to_hours
from torch.utils.data import Dataset
from compress.utils.plotting import plot_sos
from PIL import Image


def from_state_dict(cls, state_dict):

    net = cls(192, 320)
    net.load_state_dict(state_dict)
    return net

class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = [os.path.join(self.data_dir,f) for f in os.listdir(self.data_dir)]

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        #transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(image)

    def __len__(self):
        return len(self.image_path)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-m","--model",default="cnn",choices=models.keys(),help="Model architecture (default: %(default)s)",)
    parser.add_argument("-aux","--aux_net",default="none",choices=aux_net_models.keys(),help="Model architecture (default: %(default)s)")
    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    parser.add_argument("-e","--epochs",default=300,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument("-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",)
    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--lmbda",type=float,default=0.011, help="Bit-rate distortion parameter (default: %(default)s)",)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size",type=int,default=64,help="Test batch size (default: %(default)s)",)
    parser.add_argument( "--aux-learning-rate", default=1e-3, type=float, help="Auxiliary loss learning rate (default: %(default)s)",)
    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--save_path", type=str, default="7ckpt/model.pth.tar", help="Where to Save model")
    parser.add_argument("--seed", type=float,default = 42, help="Set random seed for reproducibility")
    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")

    parser.add_argument("-ni","--num_images",default = 8016, type = int)
    parser.add_argument("-niv","--num_images_val",default = 1024, type = int)

    parser.add_argument("-lt","--loss_type",default = "mse", type = str)
    parser.add_argument("-dims","--dimension",default=192,type=int,help="Number of epochs (default: %(default)s)",) 
    parser.add_argument("-dims_m","--dimension_m",default=320,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument("-q","--quality",default=3,type=int,help="Number of epochs (default: %(default)s)",)

    parser.add_argument("-he","--halve_epoch",default=10,type=int,help="Number of epochs (default: %(default)s)",)


    parser.add_argument("--fact_beta",default=10,type=float,help="factorized_beta",)
    parser.add_argument("--fact_num_sigmoids",default =0,type=int,help="factorized_beta",)
    parser.add_argument("--fact_extrema",default=20,type=int,help="factorized_extrema",)
    parser.add_argument("--fact_gp",default=15,type=int,help="factorized_beta",)
    parser.add_argument("--fact_activation",default="nonlinearstanh",type=str,help="factorized_beta",)
    parser.add_argument("--fact_annealing",default="gap_stoc",type=str,help="factorized_annealing",)
    parser.add_argument("--fact_tr",default= True,type=bool,help="factorized_tr",)



    parser.add_argument("--gauss_beta",default=10,type=float,help="gauss_beta",)
    parser.add_argument("--gauss_num_sigmoids",default=0,type=int,help="gauss_beta",)
    parser.add_argument("--gauss_extrema",default=60,type=int,help="gauss_extrema",)
    parser.add_argument("--gauss_gp",default=15,type=int,help="gauss_beta",)
    parser.add_argument("--gauss_activation",default="nonlinearstanh",type=str,help="factorized_beta",)
    parser.add_argument("--gauss_annealing",default="gap_stoc",type=str,help="factorized_annealing",)
    parser.add_argument("--gauss_tr",default=True,type=bool,help="gauss_tr",)

    parser.add_argument("--baseline", default=False, type=bool, help="factorized_annealing",)
    parser.add_argument("--classic_compress", default= True, type=bool, help="compress_classical",)
    parser.add_argument("--filename",default="/data/",type=str,help="factorized_annealing",)
    parser.add_argument("--suffix",default=".pth.tar",type=str,help="factorized_annealing",)

    parser.add_argument("--pret_checkpoint",default = "/scratch/pretrained_models/devil2022/q3-zou22.pth.tar")
    
    parser.add_argument("--pret_checkpoint_base",default ="/scratch/pretrained_models/devil2022/q3-zou22.pth.tar") # "/scratch/pretrained_models/BlockAttention/base_cnn_0483.pth.tar"

    parser.add_argument("--pret_checkpoint_stf",default ="/scratch/pretrained_models/stf/stf_0483.pth.tar") # "/scratch/pretrained_models/stf/stf_013.pth.tar"
    parser.add_argument("--path_adapter",default = "/scratch/inference/new_models/devil2022/4anchors/q6-zou22.pth.tar" ) 
    parser.add_argument("--adapt",default = True, type = bool)
    
    # /devil2022  ddd



    args = parser.parse_args(argv)
    return args

def rename_key(key):
    """Rename state_dict key.rrr"""

    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]
    if key.startswith('h_s.'):
        return None

    # ResidualBlockWithStride: 'downsample' -> 'skip'dd
    # if ".downsample." in key:
    #     return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters  pppp
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key



def load_pretrained(state_dict):
    """Convert sccctaddte_dict keys."""
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    if None in state_dict:
        state_dict.pop(None)
    return state_dict


def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])


def modify_dictionary(check):
    res = {}
    ks = list(check.keys())
    for key in ks: 
        res[key[7:]] = check[key]
    return res


import time
def main(argv):
    args = parse_args(argv)
    print(args,"cc")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )


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



    factorized_configuration , gaussian_configuration = configure_latent_space_policy(args)
    print("gaussian configuration----- -fddffdguuggfffffdddxtttssssxxx------>: ",gaussian_configuration)
    print("factorized configuration------>cccccàsssààcccc->: ",factorized_configuration)
    if args.baseline is False:
        annealing_strategy_bottleneck, annealing_strategy_gaussian =  configure_annealings(factorized_configuration, gaussian_configuration)
    else:
        annealing_strategy_bottleneck, annealing_strategy_gaussian =  None, None




    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)



    aux_net = aux_net_models[ args.aux_net]
    if aux_net is not None:
        #aux_net = aux_net()
        #print("prima di fare l'update abbiamo che: ",aux_net.g_a[0].weight[0])
        print("Loading", args.pret_checkpoint_base)           
        #checkpoint = torch.load(args.pret_checkpoint, map_location=device)
        state_dict = load_pretrained(torch.load(args.pret_checkpoint_base, map_location=device)['state_dict'])
        aux_net = from_state_dict(aux_net_models["cnn"], state_dict)
        print("DOPO: ",aux_net.g_a[0].weight[0])
        #aux_net = from_state_dict(aux_net,checkpoint["state_dict"])
        aux_net.update(force = True)
        aux_net.to(device) 


    N = args.dimension
    M = args.dimension_m
    if args.model == "cnn":# and args.baseline is False :
        if args.path_adapter == "":
            print("io entro qua, è la cosa guysta!")
            net = models[args.model](N = N, M = M, factorized_configuration = factorized_configuration, gaussian_configuration = gaussian_configuration , pretrained_model = aux_net)
            sos = True
        else: 
            print("qua ci entro se faccio adapter o riprendo il training!!!!")
            architecture =   models[args.model]
            checkpoint = torch.load(args.path_adapter, map_location=device)

            factorized_configuration =checkpoint["factorized_configuration"]
            gaussian_configuration =  checkpoint["gaussian_configuration"]
            if args.adapt:
                factorized_configuration["beta"] = 10
                factorized_configuration["trainable"] = True
                factorized_configuration["annealing"] = args.fact_annealing
                factorized_configuration["gap_factor"] = args.fact_gp


                
                gaussian_configuration["beta"] = 10
                gaussian_configuration["trainable"] = True
                gaussian_configuration["annealing"] = args.gauss_annealing
                gaussian_configuration["gap_factor"] = args.gauss_gp

            net =architecture(192, 320, factorized_configuration = factorized_configuration, gaussian_configuration = gaussian_configuration)
            net = net.to(device)              
            net.update( device = device)
            net.load_state_dict(checkpoint["state_dict"])  
            print("**************************************************************************************************************") 
            print("**************************************************************************************************************")  
            net.entropy_bottleneck.sos.update_state(device = device )
            net.gaussian_conditional.sos.update_state(device = device)
            print("weightsss!!!!- ",net.gaussian_conditional.sos.cum_w)
            
            print("************** ho finito il caricamento!***********************************************************************************************")  
            print("**************************************************************************************************************")  

            net.update( device = device)
            sos = True


    elif args.model == "stanh_cnn":

        net = models[args.model](N = N, M = M, factorized_configuration = factorized_configuration, gaussian_configuration = gaussian_configuration)
        checkpoint = torch.load(args.pret_checkpoint, map_location=device)
        print("prima di fare l'update abbiamo che: ",net.h_a[0].weight[0])
        net.load_state_dict(checkpoint["state_dict"])
        print("DOPOOOOOOOO---->",net.g_a[0].weight[0])
        sos = True       




    else:
        net = models[args.model]( )
        sos = False
    net = net.to(device)


    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)



    criterion = RateDistortionLoss(lmbda=args.lmbda)
    

    last_epoch = 0



    optimizer, aux_optimizer = configure_optimizers(net, args)
    print("hola!")
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=50)


    counter = 0
    best_loss = float("inf")
    epoch_enc = 0





    previous_lr = optimizer.param_groups[0]['lr']
    print("subito i paramteri dovrebbero essere giusti!")
    model_tr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    model_fr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad== False)
        
    print(" Ttrainable parameters: ",model_tr_parameters)
    print(" freeze parameters: ", model_fr_parameters)

    #net.unfreeze_quantizer
    #net.print_pars()
    fact_gp = args.fact_gp
    gauss_gp = args.gauss_gp



    if args.adapt:
        net.freeze_net() 
        print("start unfreezing")
        net.unfreeze_quantizer()



    model_tr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    model_fr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad== False)
    print("******************************* DOPO")
    print(" trainable parameters: ",model_tr_parameters)
    print(" freeze parameters: ", model_fr_parameters)


    for epoch in range(last_epoch, args.epochs):
        print("**************** epoch: ",epoch,". Counter: ",counter)
        previous_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}","    ",previous_lr)
        print("epoch ",epoch)
        start = time.time()
        counter = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            counter,
            annealing_strategy_bottleneck, 
            annealing_strategy_gaussian,
            sos
        )


        loss_valid = test_epoch(epoch, valid_dataloader, net, criterion, sos, valid = True)

        loss = test_epoch(epoch, test_dataloader, net, criterion, sos, valid = False)

        lr_scheduler.step(loss_valid)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        filename, filename_best =  create_savepath(args, epoch)
        if args.baseline: #and (is_best or epoch%25==0):
            if  (is_best or epoch%25==0):
                save_checkpoint_our(
                    {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                filename,
                filename_best
                )
        else:

            filename, filename_best =  create_savepath(args, epoch)

            if (is_best) or epoch%5==0:
                if sos:
                    save_checkpoint_our(
                            {
                                "epoch": epoch,
                                "annealing_strategy_bottleneck":annealing_strategy_bottleneck,
                                "annealing_strategy_gaussian":annealing_strategy_gaussian,
                                "state_dict": net.state_dict(),
                                "loss": loss,
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "factorized_configuration": net.factorized_configuration,
                                "gaussian_configuration":net.gaussian_configuration,
                       #         "N": net.N,
                       #         "M": net.M,
                                "entropy_bottleneck_w":net.entropy_bottleneck.sos.w,
                                "entropy_bottleneck_b":net.entropy_bottleneck.sos.b,

                        },
                        is_best,
                        filename,
                        filename_best    
                    )   
                else:
                    save_checkpoint_our(
                            {
                                "epoch": epoch,
                                "state_dict": net.state_dict(),
                                "loss": loss,
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),

                        },
                        is_best,
                        filename,
                        filename_best    
                    )   



        print("log also the current leraning rate")

        log_dict = {
        "train":epoch,
        "train/leaning_rate": optimizer.param_groups[0]['lr'],
        #"train/beta": annealing_strategy_gaussian.beta
        }

        wandb.log(log_dict)
        # learning rate stuff 
        print("start the part related to beta")
        
        
        #if (epoch + 1)%1==0:
        #annealing_strategy_bottleneck, annealing_strategy_gaussian =  configure_annealings(factorized_configuration, gaussian_configuration)
        #else:
        #print("not updating the beta")



        """
        if (epoch+1)%ddargs.halve_epoch==0:
            print("divrrrido il max beta")
            factorized_configuration["beta"] = annealing_strategy_bottleneck.beta/10
            gaussian_configuration["beta"] = annealing_strategy_gaussian.beta/10 
            net.entropy_bottleneck.sos.beta = annealing_strategy_bottleneck.beta/10 
            net.gaussian_conditional.sos.beta = annealing_strategy_gaussian.beta/10
            annealing_stratdddegy_bottleneck, annealing_strategy_gaussian =  configure_annealings(factorized_configuration, gaussian_configuration)
        print(annealingd_sssstrategy_gaussian.max_beta)
        """

        
        
        if epoch%2==0:
            print("entro qua")


            # create filepath 
            #filelist = [os.path.join("/scratch/dataset/kodak", f) for f in os.listdir("/scratch/dataset/kodak")]
            net.update()
            #epoch_enc += 1
            #compress_with_ac(net, filelist, device, epoch_enc, baseline = args.baseline) ddddd
            if args.baseline is False and sos: 
                plot_sos(net, device)
        
        
        
        if sos:

            if factorized_configuration is not None and  factorized_configuration["annealing"] == "random_gp" and annealing_strategy_bottleneck.right_beta <= 10000:
                print("entro qua")
                annealing_strategy_bottleneck.right_beta += fact_gp 
        

            if  gaussian_configuration is not None and gaussian_configuration["annealing"] == "random_gp" and annealing_strategy_gaussian.right_beta <= 10000:
                print("entro qu   aaaa")
                annealing_strategy_gaussian.right_beta += gauss_gp
            if gaussian_configuration is not None and gaussian_configuration["annealing"] == "triangle":
                if  annealing_strategy_gaussian.increase:
                    annealing_strategy_gaussian.increase = False
                    print("now the annealing is: ", annealing_strategy_gaussian.increase)
                else:
                    annealing_strategy_gaussian.increase = True
                    print("now the annealing is: ", annealing_strategy_gaussian.increase)
            end = time.time()
            print("Runtime of the epoch  ", epoch)
            sec_to_hours(end - start) 
            print("END OF EPOCH ", epoch)

            if args.gauss_activation=="delta" and args.baseline is False:

                log_dict = {
                "train":epoch,
                "train/gaussiandelta": net.gaussian_conditional.sos.delta.data.item()
                }

                wandb.log(log_dict)   

            if args.fact_activation=="delta" and args.baseline is False:

                log_dict = {
                "train":epoch,
                "train/factorized_delta": net.entropy_bottleneck.sos.delta.data.item()
                }

                wandb.log(log_dict)   

            if optimizer.param_groups[0]['lr']!=previous_lr and gaussian_configuration["annealing"] == "gap_stoc": #
                annealing_strategy_gaussian.factor = annealing_strategy_gaussian.factor*2 
                annealing_strategy_bottleneck.factor = annealing_strategy_bottleneck.factor*2 
            elif optimizer.param_groups[0]['lr']!=previous_lr and gaussian_configuration["annealing"] == "gap_stoc":
                fact_gp = fact_gp*2 
                gauss_gp = gauss_gp*2
      
        

if __name__ == "__main__":
    wandb.init(project="NeuralADQ_zou2022_A2_sections", entity="albertopresta")   
    main(sys.argv[1:])




