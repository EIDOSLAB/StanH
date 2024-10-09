import torch 
import torch.nn as nn
import wandb
from compress.utils.help_function import compute_msssim, compute_psnr
from compressai.ops import compute_padding
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F

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


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm ,counter, annealing_strategy_entropybottleneck , annealing_strategy_gaussian, sos ):
    model.train()
    device = next(model.parameters()).device



    
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()


    y_bpp = AverageMeter()
    z_bpp = AverageMeter()

    fact_beta = AverageMeter()
    gauss_beta = AverageMeter()


    for i, d in enumerate(train_dataloader):
        counter += 1
        d = d.to(device)

        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()

        if sos:
            out_net = model(d, training = True)
            gap = out_net["gap"]
        else: 
            out_net = model(d)



        out_criterion = criterion(out_net, d)
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


        if "z_bpp" in list(out_criterion.keys()):

            wand_dict = {
                "train_batch": counter,
                "train_batch/factorized_bpp": out_criterion["z_bpp"].clone().detach().item(),
                "train_batch/gaussian_bpp": out_criterion["y_bpp"].clone().detach().item()

            }
            wandb.log(wand_dict)  
            y_bpp.update(out_criterion["y_bpp"].clone().detach())
            z_bpp.update(out_criterion["z_bpp"].clone().detach())             
        # we have to augment beta here 
        

        if sos:
            wand_dict = {
                "general_data/":counter,
                "general_data/factorized_gap: ": gap[0]
            }
            
            wandb.log(wand_dict)

        if "z_bpp" in list(out_criterion.keys()):
            if sos:
                wand_dict = {
                    "general_data":counter,
                    "general_data/gaussian_gap: ": gap[1]
                }
                
                wandb.log(wand_dict)
            
        

        if sos:
            if  annealing_strategy_entropybottleneck is not  None:

            
                if annealing_strategy_entropybottleneck.type == "triangle":
                    annealing_strategy_entropybottleneck.step(gap = gap[0])
                    model.entropy_bottleneck.sos.beta = annealing_strategy_entropybottleneck.beta
                elif "random" in annealing_strategy_entropybottleneck.type:
                    annealing_strategy_entropybottleneck.step(gap = gap[0])
                    model.entropy_bottleneck.sos.beta = annealing_strategy_entropybottleneck.beta
        
                else:
                    
                    lss = out_criterion["loss"].clone().detach().item()
                    annealing_strategy_entropybottleneck.step(gap[0], epoch, lss)
                    model.entropy_bottleneck.sos.beta = annealing_strategy_entropybottleneck.beta

                fact_beta.update(annealing_strategy_entropybottleneck.beta)

            if annealing_strategy_gaussian is not None:
                if annealing_strategy_gaussian.type == "triangle":
                    annealing_strategy_gaussian.step(gap = gap[1])
                    model.gaussian_conditional.sos.beta = annealing_strategy_gaussian.beta
                elif "random" in annealing_strategy_gaussian.type:
                    annealing_strategy_gaussian.step(gap = gap[1])
                    model.gaussian_conditional.sos.beta = annealing_strategy_gaussian.beta
                else:
                    lss = out_criterion["loss"].clone().detach().item()
                    annealing_strategy_gaussian.step(gap[1], epoch, lss)
                    model.gaussian_conditional.sos.beta = annealing_strategy_gaussian.beta


                wand_dict = {
                    "general_data/":counter,
                    "general_data/gaussian_beta: ": model.gaussian_conditional.sos.beta
                }
                
                wandb.log(wand_dict)

                gauss_beta.update(model.gaussian_conditional.sos.beta)

    log_dict = {
        "train":epoch,
        "train/loss": loss.avg,
        "train/bpp": bpp_loss.avg,
        "train/mse": mse_loss.avg,
        "train/gauss_beta": gauss_beta.avg
        }
        
    wandb.log(log_dict)



    if "z_bpp" in list(out_criterion.keys()):
        if sos:
            wand_dict = {
                "train":epoch,
                "train/factorized_bpp": z_bpp.avg,
                "train/gaussian_bpp": y_bpp.avg,
                "train/gaussian_beta":gauss_beta.avg

            }
            wandb.log(wand_dict)





    return counter




def test_epoch(epoch, test_dataloader, model, criterion, sos, valid):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()


    psnr = AverageMeter()
    ssim = AverageMeter()
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            if sos:
                out_net = model(d, training = False)
            else: 
                out_net = model(d)

            out_criterion = criterion(out_net, d)

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


def read_image(filepath, clic =False):
    #assert filepath.is_file()
    img = Image.open(filepath)
    
    if clic:
        i =  img.size
        i = i[0]//2, i[1]//2
        img = img.resize(i)
    img = img.convert("RGB")
    return transforms.ToTensor()(img)



def compress_with_ac(model, filelist, device, epoch, baseline = False):
    #model.update(None, device)
    print("ho finito l'update")
    bpp_loss = AverageMeter()
    psnr = AverageMeter()
    mssim = AverageMeter()

    
    with torch.no_grad():
        for i,d in enumerate(filelist): 
            if baseline is False:
                print("-------------    ",i,"  --------------------------------")
                x = read_image(d).to(device)
                x = x.unsqueeze(0) 
                h, w = x.size(2), x.size(3)
                pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
                x_padded = F.pad(x, pad, mode="constant", value=0)


                #data = model.compress(x_padded)
                print("shape: ",x_padded.shape)
                out_enc = model.compress(x_padded)
                #out_net = model(x_padded,  training = False)
                #out_dec = model.decompress(data)
                out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
                out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
                out_net["x_hat"] = F.pad(out_dec["x_hat"], unpad)




                out_dec["x_hat"].clamp_(0.,1.)
                out_net["x_hat"].clamp(0.,1.)
                

                bpp, bpp_1, bpp_2= bpp_calculation(out_dec, data["strings"])
                bpp_loss.update(bpp)
                psnr.update(compute_psnr(x, out_dec["x_hat"]))

                mssim.update(compute_msssim(x, out_net["x_hat"]))   
                print("bpp---> ",bpp,"  ",bpp_1,"   ",bpp_2) 

                    
                xhat = out_net["x_hat"].ravel()
                xcomp = out_dec["x_hat"].ravel()
                for i in range(10):
                    print(xhat[i],"---", xcomp[i]) 
            else:
                #out_enc = model.compress(d)
                d = d.to(device)
                out_enc = model.compress_classical(d)
                out_net = model(d, training = False)
                out_dec = model.decompress_classical(out_enc["strings"], out_enc["shape"])
                #out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
                out_dec["x_hat"].clamp_(0.,1.)
                out_net["x_hat"].clamp(0.,1.)

                
                num_pixels = d.size(0) * d.size(2) * d.size(3)
                bpp =  bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels#, bpp_1, bpp_2= bpp_calculation(out_dec, out_enc["strings"])
                bpp_loss.update(bpp)
                psnr.update(compute_psnr(d, out_dec["x_hat"]))

                mssim.update(compute_msssim(d, out_net["x_hat"]))   

                    
                xhat = out_net["x_hat"].ravel()
                xcomp = out_dec["x_hat"].ravel()
                for i in range(10):
                    print(xhat[i],"---", xcomp[i]) 
            




    log_dict = {
            "test":epoch,
            "test/bpp_with_ac": bpp_loss.avg,
            "test/psnr_with_ac": psnr.avg,
            "test/mssim_with_ac":mssim.avg
    }
    
    wandb.log(log_dict)
    return bpp_loss.avg


def bpp_calculation(out_net, out_enc):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]

        bpp_1 = (len(out_enc[0]) * 8.0 ) / num_pixels
        bpp_2 =  sum( (len(out_enc[1][i]) * 8.0 ) / num_pixels for i in range(len(out_enc[1])))
        return bpp_1 + bpp_2, bpp_1, bpp_2


