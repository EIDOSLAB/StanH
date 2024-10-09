import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
import numpy as np
#import wandb
import pandas as pd 


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys 
import copy
import os 

def evaluation(model,filelist,entropy_estimation,device,epoch = -10):



    levels = [i for i in range(model.num_stanh)]

    psnr = [AverageMeter() for _ in range(model.num_stanh)]
    ms_ssim = [AverageMeter() for _ in range(model.num_stanh)]
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

            clear_memory()
        
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

def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])


def clear_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


class NonLinearStanh(nn.Module):
    def __init__(self,
                  beta,  
                  num_sigmoids, 
                    extrema = 5, 
                    list_random_w = [],
                    list_random_b = [],
                    trainable =False):
        super(NonLinearStanh, self).__init__()
        print("non-linear-sum")
        self.num_sigmoids = int(num_sigmoids)
        self.beta = beta
        self.extrema = extrema     
        self.minimo = - extrema 
        self.massimo = extrema
            
        
        self.range_num = torch.arange(self.minimo  + 0.5 ,self.massimo ).type(torch.FloatTensor)
        if self.num_sigmoids > 0:
            self.jump = len(self.range_num)/self.num_sigmoids
            self.levels = num_sigmoids + 1
        
        else:
            self.levels = extrema*2 + 1 


        # bias 
        if self.num_sigmoids == 0:
            if list_random_b == []:
                self.b = torch.nn.Parameter(self.range_num.type(torch.FloatTensor), requires_grad= trainable) # + torch.tensor([0.32,-0.15, 0.23, 0.05])#+ torch.relu(torch.randn(len(self.range_num))) # quantizzazione allenabile (ha senso)?
            else:
                self.b = torch.nn.Parameter(self.range_num.type(torch.FloatTensor), requires_grad= trainable) + torch.tensor(list_random_b)
        else:
                #self.b = torch.nn.Parameter(torch.FloatTensor(num_sigmoids).normal_().sort()[0]) # punti a caso
            c = len(self.range_num)/self.num_sigmoids
            self.b = torch.nn.Parameter(torch.arange(self.minimo + self.jump/2   ,self.massimo + self.jump/2 , c),  requires_grad= trainable)


        if self.num_sigmoids == 0:
            if list_random_w == []:
                self.w = torch.nn.Parameter(torch.ones(len(self.range_num)), requires_grad= trainable ) #+ torch.tensor([0.23,0.24, 0.20, 0.27])
            else:
                self.w = torch.nn.Parameter(torch.ones(len(self.range_num)), requires_grad= trainable ) + torch.tensor(list_random_w)
        else:
            self.w = torch.nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump ,  requires_grad= trainable )      


    
        self.tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        #print("trainable parameters for the quantizer: ",self.tr_parameters)


        self.length = len(self.range_num) if self.num_sigmoids ==0 else self.num_sigmoids 
        #print("lunghezza---> ",self.length)

        self.map_sos_cdf = {}
        self.map_cdf_sos = {}




        n = (torch.sum(self.w)/2).item()
        self.cum_w = torch.zeros(self.length + 1)
        self.cum_w[1:] = torch.cumsum(self.w,dim = 0)  
        self.cum_w = torch.sub(self.cum_w,n)
        #print("CUMULATIVE WEIGHTS ARE: ",self.cum_w)


        self.calculate_average_points()
        self.calculate_distance_points()

        self.update_state()

        


    def update_state(self, device = torch.device("cpu")):
        self.update_cumulative_weights(device = device )
        self.calculate_average_points( ) #self.average_points
        self.average_points = self.average_points.to(device)
        self.calculate_distance_points() #self.distance_points
        self.distance_points = self.distance_points.to(device)
        self.define_channels_map()



    def calculate_average_points(self):
        self.average_points = torch.add(self.cum_w[1:], self.cum_w[:-1])/2
        

    def calculate_distance_points(self):
        self.distance_points = torch.sub(self.cum_w[1:], self.cum_w[:-1])/2
        


    def update_cumulative_weights(self,device = torch.device("cpu")):
        #if self.num_sigmoids == 0:
        n = (torch.sum(self.w)/2).item()
        self.cum_w = torch.zeros(self.length + 1).to(self.w.device)
        self.cum_w[0] = 0.0
        self.cum_w[1:] = torch.cumsum(self.w,dim = 0)
        self.cum_w = torch.sub(self.cum_w,n) # -  self.extrema 
        self.cum_w = self.cum_w.to(device)


    def reinitialize_weights_and_bias(self):
        if self.num_sigmoids == 0:
            self.w = torch.nn.Parameter(torch.ones(len(self.range_num)) )
            self.b = torch.nn.Parameter(self.range_num.type(torch.FloatTensor))
        else:
            self.w = torch.nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump  ) 
            c = len(self.range_num)/self.num_sigmoids
            self.b = torch.nn.Parameter(torch.arange(self.minimo + self.jump/2   ,self.massimo + self.jump/2 , c))
        






    def define_channels_map(self):
        mapping = torch.arange(0, int(self.cum_w.shape[0]), 1).numpy()
        map_float_to_int = dict(zip(list(self.cum_w.detach().cpu().numpy()),list(mapping)))
        map_int_to_float = dict(zip(list(mapping),list(self.cum_w.detach().cpu().numpy())))            
        self.map_sos_cdf = map_float_to_int
        self.map_cdf_sos = map_int_to_float

        #print("maps: ",self.map_sos_cdf)

        #self.mapping_decoding = pd.DataFrame(list(self.map_cdf_sos.items()), columns=['key', 'value'])
        #self.mapping_decoding.set_index('key', inplace=True)
        #print("***************************** mapping fatto")
    
    


    def f(self,x):
        return 2*torch.sigmoid(2*x) - 1

    def forward(self, x, beta=None):
        #if self.trainable_bias:
        b = torch.sort(self.b)[0] # non serve ?
        #else:
        #    b = self.b   
         
        if beta is not None:
            if beta == -1:
                return torch.sum(self.w[:,None]*torch.relu(torch.sign(x - b[:,None])) - self.w[:,None]/2,dim = 1).unsqueeze(1)             
                #return torch.stack([self.w[i]*(torch.relu(torch.sign(x-b[i]))) - self.w[i]/2 for i in range(self.length)], dim=0).sum(dim=0) 
            else:
                #return torch.sum((self.w[:,None]/2)*self.f(beta*(x - b[:,None].to(x.device))),dim = 1).unsqueeze(1)
                return torch.stack([(self.w[i]/2)*self.f(beta*(x-b[i])) for i in range(self.length)], dim=0).sum(dim=0) 
        else:
            return torch.sum( (self.w[:,None]/2)* self.f(self.beta*(x - b[:,None]))  ,dim = 1).unsqueeze(1)
            #return torch.stack([(self.w[i]/2)*self.f(self.beta*(x-b[i])) for i in range(self.length)], dim=0).sum(dim=0)

        


num_sigmoids = 0
beta= 20
beta2 = 0.1
extrema = 60
x = torch.arange(-5.5,5.5,0.01)


xx = x.repeat(192,1,1)#.reshape(192,1,1200)
print(xx.shape)
#print("---->",x.shape)


nums = int(extrema*4)
sumt =  NonLinearStanh(beta,nums,extrema = extrema)
sumt2 =  NonLinearStanh(beta,nums,extrema = extrema, )
sumt3 = NonLinearStanh(beta, nums,extrema = extrema)
sumt4 = NonLinearStanh(beta2,nums,extrema = extrema)

sumt3.w = torch.nn.Parameter(sumt2.w) #torch.nn.Parameter((sumt2.w*0.99+ sumt.w*0.01))
sumt3.b = torch.nn.Parameter((sumt2.b*0.99 + sumt.b*0.01)) #torch.nn.Parameter(sumt2.b) #
sumt3.update_state()

sumt4.w = torch.nn.Parameter(sumt2.w) #torch.nn.Parameter((sumt2.w*0.5+ sumt.w*0.5))
sumt4.b =  torch.nn.Parameter((sumt2.b*0.5 + sumt.b*0.5)) #torch.nn.Parameter(sumt2.b) #
sumt4.update_state()


print("---------------------------------------> B",sumt.b)
print("************************** W: ",sumt.w )
print("cumulative weights",sumt.cum_w)




"""

c = sumt(xx,-1)
print(torch.max(c))
print(c.shape)

w = torch.tensor([-1.23, 0.18, 0.87])
mean_p = torch.add(w[1:], w[:-1])/2
t = torch.tensor([1,1])
print(t,"    ", mean_p)
def quantize(x, w, b, inverse = False): 
    if inverse is False:
        return torch.sum(torch.relu((w[:,None].to(x.device))*(torch.sign(x - b[:,None].to(x.device)))),dim = 1).unsqueeze(1).to(x.device)   
    else:
         return torch.sum((w[:,None].to(x.device)/2)*(torch.sign(x - b[:,None].to(x.device))),dim = 1).unsqueeze(1).to(x.device)



maps = torch.arange(sumt.w.shape[0])
print("maps: ",maps)


import seaborn as sns
# Imposta la palette "tab10" di Seaborn
palette = sns.color_palette("tab10")
#rc('text', usetex=True)

rc('font', family='Times New Roman')


print("---------")
"""

from matplotlib import rc
import seaborn as sns
# Imposta la palette "tab10" di Seaborn
palette = sns.color_palette("tab10")
#rc('text', usetex=True)

rc('font', family='Times New Roman')

print(sumt.w)

b1 =  sumt(xx,beta = 1)
b3 = sumt(xx,beta = 3)
difference = sumt(xx,beta = 1) - sumt(xx,beta = 3)
bprova = sumt(xx,beta = 1) + 2*sumt(xx,beta = 3)
plt.figure(figsize=(12,10))
#plt.plot(xx[0,0,:].detach().numpy(), F.tanh(xx)[0,0].detach().numpy(), label='tanh')
#plt.plot(xx[0,0,:].detach().numpy(),xx[0,0,:].detach().numpy() , label='linear')
#plt.plot(xx[0,0,:].detach().numpy(),  sumt(xx,beta = 1)[0,0].detach().numpy() , label=r'$\beta = 1$')
#plt.plot(xx[0,0,:].detach().numpy(), sumt(xx,2)[0,0,:].detach().numpy(), label=r'$\beta = 2$')
#plt.plot(xx[0,0,:].detach().numpy(),  (sumt(xx,beta = 5)[0,0]).detach().numpy()  , label=r'$\beta = 5$')#
#plt.plot(xx[0,0].detach().numpy(), sumt(xx,10)[0,0].detach().numpy(), label=r'$\beta = 10$')
#plt.plot(xx[0,0].detach().numpy(), sumt(xx,20)[0,0].detach().numpy(), label=r'$\beta = 20$')
plt.plot(xx[0,0,:].detach().numpy(), sumt(xx,beta = -1)[0,0].detach().numpy() ,color = 'k', label=r'$\beta \rightarrow +\infty$')
plt.plot(xx[0,0,:].detach().numpy(), sumt2(xx,beta = -1)[0,0].detach().numpy() ,color = 'r', label=r'$2$')
plt.plot(xx[0,0,:].detach().numpy(), sumt3(xx,beta = -1)[0,0].detach().numpy() ,color = 'b', label=r'$3$')
plt.plot(xx[0,0,:].detach().numpy(), sumt4(xx,beta = -1)[0,0].detach().numpy() ,color = 'g', label=r'$4$')
#plt.plot(xx[0,0].detach().numpy() ,bprova[0,0].detach().numpy() , label='beta = {}'.format(7))
plt.grid()
plt.legend(fontsize = 18)
plt.xlabel(r'original latent representation', fontsize = 25)
plt.ylabel(r'discrete/soft-quantized latent representation', fontsize = 25)
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
plt.show()
#plt.savefig("scalini/scalini.pdf", dpi=200, bbox_inches='tight', pad_inches=0.01)
#plt.savefig()


def quantize(x, w, b): 
    return torch.sum((w[:,None].to(x.device)/2)*(torch.sign(x - b[:,None].to(x.device))),dim = 1).unsqueeze(1).to(x.device)       

