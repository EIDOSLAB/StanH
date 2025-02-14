from .stf import SymmetricalTransFormer 
import torch.nn as nn
from compress.entropy_models import   GaussianConditionalSoS
import torch
from compress.ops import ste_round
import numpy as np
import math

class SymmetricalTransFormerStanH(SymmetricalTransFormer):
    def __init__(self,
                 pretrain_img_size=256,
                 patch_size=2,
                 in_chans=3,
                 embed_dim=48,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=4,
                 num_slices=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 frozen_stages=-1,
                 use_checkpoint=False,
                 num_stanh = 3,
                 gaussian_configuration = None):
        super().__init__(pretrain_img_size = pretrain_img_size,
                         patch_size = patch_size, 
                         in_chans = in_chans,
                         embed_dim=embed_dim,
                        depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        num_slices=num_slices,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=drop_path_rate,
                        norm_layer=norm_layer,
                        patch_norm=patch_norm,
                        frozen_stages=frozen_stages,
                        use_checkpoint=use_checkpoint)
        

        self.num_stanh = num_stanh 
        self.gaussian_configuration = gaussian_configuration
        self.gaussian_conditional = nn.ModuleList(GaussianConditionalSoS(None,
                                                            channels = 384,
                                                            beta = self.gaussian_configuration[i]["beta"], 
                                                            num_sigmoids = self.gaussian_configuration[i]["num_sigmoids"], 
                                                            activation = self.gaussian_configuration[i]["activation"],
                                                            extrema = self.gaussian_configuration[i]["extrema"], 
                                                            trainable =  self.gaussian_configuration[i]["trainable"],
                                                            device = torch.device("cuda")
                                                            ) for i in range(self.num_stanh))
        


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

    def define_permutation(self, x):
        perm = np.arange(len(x.shape)) 
        perm[0], perm[1] = perm[1], perm[0]
        inv_perm = np.arange(len(x.shape))[np.argsort(perm)] # perm and inv perm
        return perm, inv_perm    


    def forward(self, x ,stanh_level = 0, training = True):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        for i in range(self.num_layers):
            layer = self.layers[i]
            x, Wh, Ww = layer(x, Wh, Ww)

        y = x
        C = self.embed_dim * 8
        y = y.view(-1, Wh, Ww, C).permute(0, 3, 1, 2).contiguous()
        y_shape = y.shape[2:]

        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

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
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        y_hat = y_hat.permute(0, 2, 3, 1).contiguous().view(-1, Wh*Ww, C)
        for i in range(self.num_layers):
            layer = self.syn_layers[i]
            y_hat, Wh, Ww = layer(y_hat, Wh, Ww)

        x_hat = self.end_conv(y_hat.view(-1, Wh, Ww, self.embed_dim).permute(0, 3, 1, 2).contiguous())
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }



    def load_state_dict(self, state_dict, strict = True, state_dicts_stanh = None):
        super().load_state_dict(state_dict,strict = strict)
        
        if state_dicts_stanh is not None:
            for i in range(len(state_dicts_stanh)):
                self.upload_stanh_values(state_dicts_stanh[i]["state_dict"],i)