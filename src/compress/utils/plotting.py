
import torch 
import wandb




import seaborn as sns
# Imposta la palette "tab10" di Seaborn
palette = sns.color_palette("tab10")
#rc('text', usetex=True)
#rc('font', family='Times New Roman')
import matplotlib.pyplot as plt
import wandb


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




def plot_sos(model, device,n = 1000, dim = 0,aq = False):


    if aq is False:
        """
        x_min = float((min(model.entropy_bottleneck.sos.b) + min(model.entropy_bottleneck.sos.b)*0.5).detach().cpu().numpy())
        x_max = float((max(model.entropy_bottleneck.sos.b)+ max(model.entropy_bottleneck.sos.b)*0.5).detach().cpu().numpy())
        step = (x_max-x_min)/n
        x_values = torch.arange(x_min, x_max, step)
        x_values = x_values.repeat(model.entropy_bottleneck.M,1,1)
            
        print("entro qua spero!")
        y_values= model.entropy_bottleneck.sos(x_values.to(device))[0,0,:]
        data = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
        table = wandb.Table(data=data, columns = ["x", "sos"])
        wandb.log({"FactorizedSoS/SoS at dimension " + str(dim): wandb.plot.line(table, "x", "sos", title='FactorizedSoS/SoS  with beta = {}'.format(model.entropy_bottleneck.sos.beta))})
        y_values= model.entropy_bottleneck.sos(x_values.to(device), -1)[0,0,:]
        data_inf = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
        table_inf = wandb.Table(data=data_inf, columns = ["x", "sos"])
        wandb.log({"FactorizedSoS/SoS  inf at dimension " + str(dim): wandb.plot.line(table_inf, "x", "sos", title='FactorizedSoS/SoS  with beta = {}'.format(-1))})  

        """
        x_min = float((min(model.gaussian_conditional.sos.b) + min(model.gaussian_conditional.sos.b)*0.5).detach().cpu().numpy())
        x_max = float((max(model.gaussian_conditional.sos.b)+ max(model.gaussian_conditional.sos.b)*0.5).detach().cpu().numpy())
        step = (x_max-x_min)/n
        x_values = torch.arange(x_min, x_max, step)
        x_values = x_values.repeat(model.gaussian_conditional.M,1,1)
            
        y_values=model.gaussian_conditional.sos(x_values.to(device))[0,0,:]
        data = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
        table = wandb.Table(data=data, columns = ["x", "sos"])
        wandb.log({"GaussianSoS/Gaussian SoS at dimension " + str(dim): wandb.plot.line(table, "x", "sos", title='GaussianSoS/Gaussian SoS  with beta = {}'.format(model.gaussian_conditional.sos.beta))})
        y_values= model.gaussian_conditional.sos(x_values.to(device), -1)[0,0,:]
        data_inf = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
        table_inf = wandb.Table(data=data_inf, columns = ["x", "sos"])
        wandb.log({"GaussianSoS/Gaussian SoS  inf at dimension " + str(dim): wandb.plot.line(table_inf, "x", "sos", title='GaussianSoS/Gaussian SoS  with beta = {}'.format(-1))})  


