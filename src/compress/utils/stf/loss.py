import torch.nn as nn
from pytorch_msssim import ms_ssim 
import torch 
import math
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
    