
import wandb 
import random
import torch 
import math 
import torch.nn as nn
from pytorch_msssim import ms_ssim 
import torch.optim as optim
from .utils import read_image, bpp_calculation, compute_metrics, psnr, AverageMeter
from compressai.ops import compute_padding
import torch.nn.functional as F
from compress.utils.help_function import compute_msssim, compute_psnr



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
    





def train_one_epoch(model, 
                    criterion, 
                    train_dataloader,
                    optimizer,
                    epoch, 
                    clip_max_norm ,
                    counter,
                    annealing_strategy_gaussian,
                    lmbda_list = None ):
    model.train()
    device = next(model.parameters()).device

  
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

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
                    "general_data":counter,
                    "general_data/gaussian_gap: ": gap[0]
                }
                
        wandb.log(wand_dict)
            
        
        if annealing_strategy_gaussian[quality_index] is not None:
            if annealing_strategy_gaussian[quality_index].type == "triangle":
                annealing_strategy_gaussian[quality_index].step(gap = gap[0])
                model.gaussian_conditional[quality_index].sos.beta = annealing_strategy_gaussian[quality_index].beta
            elif "random" in annealing_strategy_gaussian[quality_index].type:
                annealing_strategy_gaussian[quality_index].step(gap = gap[0])
                model.gaussian_conditional[quality_index].sos.beta = annealing_strategy_gaussian[quality_index].beta
            else:
                lss = out_criterion["loss"].clone().detach().item()
                annealing_strategy_gaussian[quality_index].step(gap[0], epoch, lss)
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