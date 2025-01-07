import math
import torch
import torch.nn as nn
import numpy as np

from compress.entropy_models import  EntropyBottleneckSoS, GaussianConditionalSoS

from .utils import  update_registered_buffers
import torch.nn.functional as F
from .cnn_stanh import WACNNSoS
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))



class WACNNMultiSTanH(WACNNSoS):
    """CNN based model"""

    def __init__(self,
                 num_stanh = 3,
                  N=192,
                  M=320,
                  factorized_configuration = None, 
                  gaussian_configuration = None,
                  pretrained_model = None,
                 **kwargs):
        super().__init__(N = N,
                          M = M,
                            factorized_configuration=factorized_configuration,
                            gaussian_configuration=gaussian_configuration,
                            pretrained_model=pretrained_model,
                            multi = True,
                            **kwargs)


        self.num_stanh = num_stanh




        self.entropy_bottleneck = nn.ModuleList(EntropyBottleneckSoS(N, 
                                               beta = self.factorized_configuration[i]["beta"], 
                                                num_sigmoids = self.factorized_configuration[i]["num_sigmoids"], 
                                                activation = self.factorized_configuration[i]["activation"],
                                                extrema = self.factorized_configuration[i]["extrema"],
                                                trainable = self.factorized_configuration[i]["trainable"],
                                                device = torch.device("cuda") 
                                                )  for i in range(self.num_stanh) )

        self.gaussian_conditional = nn.ModuleList(GaussianConditionalSoS(None,
                                                            channels = N,
                                                            beta = self.gaussian_configuration[i]["beta"], 
                                                            num_sigmoids = self.gaussian_configuration[i]["num_sigmoids"], 
                                                            activation = self.gaussian_configuration[i]["activation"],
                                                            extrema = self.gaussian_configuration[i]["extrema"], 
                                                            trainable =  self.gaussian_configuration[i]["trainable"],
                                                            device = torch.device("cuda")
                                                            ) for i in range(self.num_stanh))
        


    def compute_gap(self, inputs, y_hat, gaussian,index, perms = None):
        values =  inputs.permute(*perms[0]).contiguous() # flatten y and call it values
        values = values.reshape(1, 1, -1) # reshape values      
        y_hat_p =  y_hat.permute(*perms[0]).contiguous() # flatten y and call it values
        y_hat_p = y_hat_p.reshape(1, 1, -1) # reshape values     
        with torch.no_grad():    
            if gaussian: 
                out = self.gaussian_conditional[index].sos(values,-1) 
            else:
                out = self.entropy_bottleneck[index].sos(values, -1)
            # calculate f_tilde:  
            f_tilde = F.mse_loss(values, y_hat_p)
            # calculat f_hat
            f_hat = F.mse_loss(values, out)
            gap = torch.abs(f_tilde - f_hat)
        return gap



    def update(self, scale_table=None,device = torch.device("cuda")):
        for i in range(self.num_stanh):
            self.entropy_bottleneck[i].update(device = device ) # faccio l'update del primo spazio latente, come factorized
            if scale_table is None:
                scale_table = get_scale_table() # ottengo la scale table 
            self.gaussian_conditional[i].update_scale_table(scale_table)
            self.gaussian_conditional[i].update(device = device)
        #print("updated entire model")
            

    def load_state_dict(self, state_dict, state_dicts_stanh = None):

        #for i in range(self.num_stanh):
        #    update_registered_buffers(
        #        self.gaussian_conditional[i],
        #        "gaussian_conditional." + str(i) ,
        #        ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
        #        state_dict,
        #    )
        super().load_state_dict(state_dict, gauss_up = False)

        if state_dicts_stanh is not None:
            for i in range(len(state_dicts_stanh)):
                self.upload_stanh_values(state_dicts_stanh[i]["state_dict"],i)
        

    def upload_stanh_values(self,state_dict,index):
        assert index < self.num_stanh

        self.gaussian_conditional[index].sos.w = torch.nn.Parameter(state_dict["gaussian_conditional"]["w"],requires_grad=True)
        self.gaussian_conditional[index].sos.b = torch.nn.Parameter(state_dict["gaussian_conditional"]["b"],requires_grad=True)
        self.entropy_bottleneck[index].sos.w = torch.nn.Parameter(state_dict["entropy_bottleneck"]["w"],requires_grad=True)
        self.entropy_bottleneck[index].sos.b = torch.nn.Parameter(state_dict["entropy_bottleneck"]["b"],requires_grad=True)

        self.gaussian_conditional[index].sos.update_state()
        self.entropy_bottleneck[index].sos.update_state()


    def unfreeze_quantizer(self,unfreeze_fact = False,indexes = None): 



        if indexes is None:
            for i in range(self.num_stanh):
                for p in self.entropy_bottleneck[i].sos.parameters(): 
                    p.requires_grad = True
                for p in self.gaussian_conditional[i].sos.parameters(): 
                    p.requires_grad = True
        else:
            for i in indexes:
                #for p in self.entropy_bottleneck[i].sos.parameters(): 
                #    p.requires_grad = True
                for p in self.gaussian_conditional[i].sos.parameters(): ##ddd
                    p.requires_grad = True  
                 
        if unfreeze_fact:
                for p in self.entropy_bottleneck[i].sos.parameters(): 
                    p.requires_grad = True             

    def unfreeze_decoder(self):
        for p in self.g_s.parameters():
            p.requires_grad = True


                



    def forward(self, x,
                 stanh_level = 0, 
                 training = True):


        self.entropy_bottleneck[math.floor(stanh_level)].sos.update_state(x.device)  # update state        
        
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        perm, inv_perm = self.define_permutation(z)
        z_hat, z_likelihoods = self.entropy_bottleneck[math.floor(stanh_level)](z, [perm,inv_perm], training = training)

        gap_entropy = self.compute_gap(z, z_hat,False, index =math.floor(stanh_level), perms = [perm, inv_perm])


        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)


        

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []


        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]


            perm, inv_perm = self.define_permutation(y)
            if stanh_level == int(stanh_level):
                self.gaussian_conditional[int(stanh_level)].sos.update_state(x.device) # update state

                y_hat_slice, y_slice_likelihood = self.gaussian_conditional[int(stanh_level)](y_slice,
                                                                                      training = training, 
                                                                                      scales = scale, 
                                                                                      means = mu, 
                                                                                      perms = [perm, inv_perm])
            else:
                floor, ceil, decimal = self.get_floor_ceil_decimal(stanh_level)
                gauss_conditional_middle = self.define_gaussian_conditional(floor, ceil,decimal)

                y_hat_slice, y_slice_likelihood = gauss_conditional_middle(y_slice,
                                                                           training = training, 
                                                                           scales = scale,
                                                                           means = mu,
                                                                           perms = [perm, inv_perm])
                
                
            y_likelihood.append(y_slice_likelihood)

            #y_hat_slice = self.gaussian_conditional.quantize(y_slice,mode = "dequantize",means = mu, perms = [perm, inv_perm]) # sos(y -mu, -1) + mu
            #y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        perm, inv_perm = self.define_permutation(y)
        
        if stanh_level == int(stanh_level):
            y_gap = self.gaussian_conditional[int(stanh_level)].quantize(y, "training" if training else "dequantize", perms = [perm, inv_perm])                           
            gap_gaussian =  self.compute_gap(y,  y_gap, True, index = int(stanh_level),perms =  [perm, inv_perm])
        else: 
            y_gap = self.gaussian_conditional[math.floor(stanh_level)].quantize(y, "training" if training else "dequantize", perms = [perm, inv_perm])                           
            gap_gaussian =  self.compute_gap(y,  y_gap, True, index = math.floor(stanh_level),perms =  [perm, inv_perm])



        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "gap":[gap_entropy, gap_gaussian]
        }


    def get_floor_ceil_decimal(self,num):
        floor_num = math.floor(num)
        ceil_num = math.ceil(num)
        decimal_part = num - floor_num
        return floor_num, ceil_num, decimal_part
    


    def define_gaussian_conditional(self,floor,ceil,decimal):

        first_sos = self.gaussian_conditional[floor].sos
        second_sos = self.gaussian_conditional[ceil].sos 

        custom_w = first_sos.w*(1-decimal) + second_sos.w*decimal 
        custom_b = first_sos.b*(1-decimal) + second_sos.b*decimal 

        gaussian_cond = self.gaussian_conditional[floor] if decimal <= 0.5 else self.gaussian_conditional[ceil] #dddd

        gaussian_cond.sos.w = torch.nn.Parameter(custom_w)
        gaussian_cond.sos.b  = torch.nn.Parameter(custom_b)
        gaussian_cond.sos.update_state()#dddd

        return gaussian_cond






    def compress(self,x,stanh_level = 0): 

        y = self.g_a(x)
        y_shape = y.shape[2:]
        #print("Y SHAPE------> ",y_shape)

        z = self.h_a(y)
    
        #al solito, tengo il vecchio modo perché non si sa mai
        z_strings, entropy_bottleneck_cdf = self.entropy_bottleneck[stanh_level].compress(z)
        z_hat = self.entropy_bottleneck[stanh_level].decompress( z_strings, entropy_bottleneck_cdf)  

        



        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)


        

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_cdfs = []
        y_shapes = []

        y_scales =  []
        y_means = []
        y_strings = []


        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional[stanh_level].build_indexes(scale)
            perm, inv_perm = self.define_permutation(y_slice)


            strings, cdfs, shapes_symb = self.gaussian_conditional[stanh_level].compress(y_slice, index,  [perm, inv_perm], means = mu) # shape is flattenend ( in theory)


            y_q_slice = self.gaussian_conditional[stanh_level].quantize(y_slice, mode = "symbols", means = mu, perms = [perm,inv_perm]) #questo va codificato!!!
            proper_shape = y_q_slice.shape

            
            y_strings.append(strings) 
            y_cdfs.append(cdfs)
            y_shapes.append(proper_shape)

            y_q_slice = self.gaussian_conditional[stanh_level].dequantize(y_q_slice) 
            y_hat_slice = y_q_slice + mu 

                                                             
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

            y_scales.append(scale)
            y_means.append(mu)
        
        return {"strings": [ z_strings, y_strings], 
                "cdfs": [ entropy_bottleneck_cdf, y_cdfs],
                "shapes": [  z.size()[-2:], y_shapes], 
                "params": {"means": y_means, "scales":y_scales}}




    def decompress(self,data,stanh_level = 0):
        strings = data["strings"] 
        cdfs = data["cdfs"]
        shapes = data["shapes"]
        z_hat = self.entropy_bottleneck[stanh_level].decompress(strings[0],cdfs[0])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[1]
        y_cdf = cdfs[1]

        y_hat_slices = []

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            #index = self.gaussian_conditional.build_indexes(scale)




            rv = self.gaussian_conditional[stanh_level].decompress(y_string[slice_index],y_cdf[slice_index]) # decompress -> dequantize  + mu
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1]).to("cuda")

            y_hat_slice = rv + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
        
        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}
    



