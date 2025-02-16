import torch
import time
from os.path import join 
from os import listdir
from torch.utils.data import DataLoader
from compress.datasets import ImageFolder
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
from compress.utils.stf.loss import RateDistortionLoss
torch.backends.cudnn.benchmark = True #sss
import wandb
import sys
import torch.optim as optim









image_models = {"zou22-base": aux_net_models["stf"],
                "zou22-sos":models["cnn_multi"],

                }



def main(argv):
    set_seed()
    args = parse_args(argv)
    device = args.device 


    wandb.init(project="STF-STANH",config = args,entity = "alberto-presta") 



    train_transforms = Compose( [RandomCrop(args.patch_size), ToTensor()])

    test_transforms = Compose([RandomCrop(args.patch_size), ToTensor()])


    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms, num_images=args.num_images)
    valid_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms, num_images=args.num_images_val)
    test_dataset = TestKodakDataset(data_dir="/scratch/dataset/kodak")


    

    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,)

    valid_dataloader = DataLoader(valid_dataset,batch_size=args.batch_size,num_workers=args.num_workers, shuffle=False,)


    test_dataloader = DataLoader(test_dataset,batch_size=1,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)

    gaussian_configuration = configure_latent_space_policy(args)
    
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
    #model.freeze_net()
    model.to("cuda")
    model.eval()
    bpp_init, psnr_init = evaluation(model,image_list,entropy_estimation = True,device = device, epoch = -10)

    print("finita valutazione iniziale: ",bpp_init," ",psnr_init) #sss

    annealing_y = []

    for ii in range(model.num_stanh):
        annealing_strategy_gaussian =  configure_annealings( gaussian_configuration)
        annealing_y.append(annealing_strategy_gaussian)
    
    optimizer = configure_optimizers(model, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=10)
    criterion = RateDistortionLoss(lmbda=args.lmbda)



    model.freeze_net()
    model.unfreeze_quantizer()
    previous_lr = optimizer.param_groups[0]['lr']
    model_tr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_fr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad== False)
    print(" trainable parameters: ",model_tr_parameters)
    print(" freeze parameters: ", model_fr_parameters)
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
                                "annealing_strategy_gaussian":annealing_strategy_gaussian,
                                "state_dict": model.state_dict(),
                                "loss": loss,
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "gaussian_configuration":model.gaussian_configuration,
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
        bpp_post, psnr_post = evaluation(model,image_list,entropy_estimation = True,device = device, epoch = epoch)
        bpp_res["our_init"] = bpp_init
        psnr_res["our_init"] = psnr_init

        bpp_res["our_post"] = bpp_post
        psnr_res["our_post"] = psnr_post


        psnr_res["base"] =      [ 30.50,32.15,33.97]
        bpp_res["base"] =      [0.191,0.298,0.441]
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



