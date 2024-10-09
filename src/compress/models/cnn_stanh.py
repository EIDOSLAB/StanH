import math
import torch
import torch.nn as nn
import numpy as np
from compressai.ans import BufferedRansEncoder, RansDecoder
from compress.entropy_models import  EntropyBottleneckSoS, GaussianConditionalSoS
from compress.layers import GDN
from .utils import conv, deconv, update_registered_buffers
from compress.ops import ste_round
from compress.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
from .base import CompressionModel
#from compress.entropy_models.adaptive_entropy_models import EntropyBottleneckSoS
#from compress.entropy_models.adaptive_gaussian_conditional import GaussianConditionalSoS
#from compress.entropy_models.adaptive_entropy_models import EntropyBottleneckSoS
import torch.nn.functional as F
from .cnn import WACNN
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class WACNNSoS(CompressionModel):
    """CNN based model"""

    def __init__(self,
                  N=192,
                  M=320,
                  factorized_configuration = None, 
                  gaussian_configuration = None,
                  pretrained_model = None,
                  multi = False,
                 **kwargs):
        super().__init__(**kwargs)



        self.N = N 
        self.M = M
        self.factorized_configuration = factorized_configuration
        self.gaussian_configuration = gaussian_configuration
        self.num_slices = 10
        self.max_support_slices = 5

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )
        self.g_s = nn.Sequential(
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, 192, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )

        self.h_scale_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
            )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )

        #self.entropy_bottleneck = EntropyBottleneck(N)
        #self.gaussian_conditional = GaussianConditional(None)





        if multi is False:
            self.entropy_bottleneck = EntropyBottleneckSoS(N, 
                                                beta = self.factorized_configuration[0]["beta"], 
                                                    num_sigmoids = self.factorized_configuration[0]["num_sigmoids"], 
                                                    activation = self.factorized_configuration[0]["activation"],
                                                    extrema = self.factorized_configuration[0]["extrema"],
                                                    trainable = self.factorized_configuration[0]["trainable"],
                                                    device = torch.device("cuda") 
                                                    )   

            self.gaussian_conditional = GaussianConditionalSoS(None,
                                                                channels = N,
                                                                beta = self.gaussian_configuration[0]["beta"], 
                                                                num_sigmoids = self.gaussian_configuration[0]["num_sigmoids"], 
                                                                activation = self.gaussian_configuration[0]["activation"],
                                                                extrema = self.gaussian_configuration[0]["extrema"], 
                                                                trainable =  self.gaussian_configuration[0]["trainable"],
                                                                device = torch.device("cuda")
                                                                )


        if  pretrained_model is not None:
            #self.replace_net(pretrained_model)
            #print("replacing the net!")
            """
            self.g_a = pretrained_model.g_a
            self.g_s = pretrained_model.g_s
            self.h_a = pretrained_model.h_a
            self.h_mean_s = pretrained_model.h_mean_s 
            self.h_scale_s = pretrained_model.h_scale_s
            self.lrp_transforms  = pretrained_model.lrp_transforms 
            self.cc_mean_transforms = pretrained_model.cc_mean_transforms 
            self.cc_scale_transforms = pretrained_model.cc_scale_transforms
            self.initialize_entropy_model(pretrained_model.entropy_bottleneck)

            """
            #print("prima di fare l'update abbiamo che: ",self.g_a[0].weight[0])
            self.initialize_bottleneck_autoencoder(pretrained_model) #g_a , g_s
            #print("finish autoencoder")
            self.initialize_entropy_model(pretrained_model.entropy_bottleneck) # nn-based neural network 
            #print("finish entropy model")
            self.initialize_hyperprior(pretrained_model) # h_a, h_mean_s, h_scale_s 
            #print("finish hyperprior")
            self.initialize_cc(pretrained_model) # cc_mean_transforms, cc_scale_transforms, lrp_transforms
            #print("finisch all")
            #print("DOPO: ",self.g_a[0].weight[0])
            
    
    def freeze_net(self):
        for n,p in self.named_parameters():
            p.requires_grad = False
        
        for p in self.parameters(): 
            p.requires_grad = False

    def unfreeze_quantizer(self): 
        for p in self.entropy_bottleneck.sos.parameters(): 
            p.requires_grad = True
        for p in self.gaussian_conditional.sos.parameters(): 
            p.requires_grad = True

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def update(self, scale_table=None,device = torch.device("cuda")):
        self.entropy_bottleneck.update(device = device ) # faccio l'update del primo spazio latente, come factorized
        if scale_table is None:
            scale_table = get_scale_table() # ottengo la scale table 
        self.gaussian_conditional.update_scale_table(scale_table)
        self.gaussian_conditional.update(device = device)
        #print("updated entire model")

    def compute_gap(self, inputs, y_hat, gaussian, perms = None):
        values =  inputs.permute(*perms[0]).contiguous() # flatten y and call it values
        values = values.reshape(1, 1, -1) # reshape values      
        y_hat_p =  y_hat.permute(*perms[0]).contiguous() # flatten y and call it values
        y_hat_p = y_hat_p.reshape(1, 1, -1) # reshape values     
        with torch.no_grad():    
            if gaussian: 
                out = self.gaussian_conditional.sos(values,-1) 
            else:
                out = self.entropy_bottleneck.sos(values, -1)
            # calculate f_tilde:  
            f_tilde = F.mse_loss(values, y_hat_p)
            # calculat f_hat
            f_hat = F.mse_loss(values, out)
            gap = torch.abs(f_tilde - f_hat)
        return gap


    def define_permutation(self, x):
        perm = np.arange(len(x.shape)) 
        perm[0], perm[1] = perm[1], perm[0]
        inv_perm = np.arange(len(x.shape))[np.argsort(perm)] # perm and inv perm
        return perm, inv_perm

    def forward(self, x, training = True):


        self.entropy_bottleneck.sos.update_state(x.device)  # update state        
        self.gaussian_conditional.sos.update_state(x.device) # update state

        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        perm, inv_perm = self.define_permutation(z)
        z_hat, z_likelihoods = self.entropy_bottleneck(z, [perm,inv_perm], training = training)

        gap_entropy = self.compute_gap(z, z_hat,False, perms = [perm, inv_perm])

        #z_hat = self.entropy_bottleneck.quantize(z,"dequantize",perms = [perm,inv_perm]) # z_hat = sos(z,-1)


        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        #z_offset = self.entropy_bottleneck._get_medians()
        #z_tmp = z - z_offset
        #z_hat = ste_round(z_tmp) + z_offset

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
            y_hat_slice, y_slice_likelihood = self.gaussian_conditional(y_slice, training = training, scales = scale, means = mu, perms = [perm, inv_perm])
            y_likelihood.append(y_slice_likelihood)

            #y_hat_slice = self.gaussian_conditional.quantize(y_slice,mode = "dequantize",means = mu, perms = [perm, inv_perm]) # sos(y -mu, -1) + mu
            #y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        perm, inv_perm = self.define_permutation(y)
        y_gap = self.gaussian_conditional.quantize(y, "training" if training else "dequantize", perms = [perm, inv_perm])
        gap_gaussian = self.compute_gap(y,  y_gap, True, perms =  [perm, inv_perm])



        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "gap":[gap_entropy, gap_gaussian]
        }

    def load_state_dict(self, state_dict,gauss_up = True):
        
        if gauss_up:
            update_registered_buffers(
                self.gaussian_conditional,
                "gaussian_conditional",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
            )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        # N = state_dict["g_a.0.weight"].size(0)
        # M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(192, 320)
        net.load_state_dict(state_dict)
        return net

    def freeze_net(self):
        for n,p in self.named_parameters():
            p.requires_grad = False
        
        for p in self.parameters(): 
            p.requires_grad = False

    def unfreeze_quantizer(self): 
        for p in self.entropy_bottleneck.sos.parameters(): 
            p.requires_grad = True
        for p in self.gaussian_conditional.sos.parameters(): 
            p.requires_grad = True



    """
    def compress(self,x): 


        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        y = self.g_a(x)
        y_shape = y.shape[2:]
        #print("Y SHAPE------> ",y_shape)

        z = self.h_a(y)
    
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress( z_strings, y_shape) 


        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)     

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []
        y_scales = []
        y_means = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]


            perm, inv_perm = self.define_permutation(y_slice)
            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, mode = "symbols", means = mu, perms = [perm,inv_perm]) #questo va codificato!!!
  

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            y_q_slice = self.gaussian_conditional.dequantize(y_q_slice) 
            y_hat_slice = y_q_slice + mu 

                                                             
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

            y_scales.append(scale)
            y_means.append(mu)
        
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:], "params": {"means": y_means, "scales": y_scales}}
    

            
    def decompress_new(self,strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]



            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv) + mu



            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
        
        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}
    """




    def initialize_bottleneck_autoencoder(self, pretrained_model):
        with torch.no_grad():
            for i,l in enumerate(pretrained_model.g_a):
                if i in [0,2,5,7]: # convolutional level 2d
                    self.g_a[i].weight = pretrained_model.g_a[i].weight
                    self.g_a[i].weight.requires_grad = True
                    self.g_a[i].bias = pretrained_model.g_a[i].bias
                    self.g_a[i].requires_grad = True
                elif i in [1,3,6]:
                    self.g_a[i].beta = pretrained_model.g_a[i].beta
                    self.g_a[i].beta.requires_grad=True
                    self.g_a[i].beta_reparam.pedestal = pretrained_model.g_a[i].beta_reparam.pedestal
                    self.g_a[i].beta_reparam.pedestal.requires_grad = True
                    self.g_a[i].gamma = pretrained_model.g_a[i].gamma
                    self.g_a[i].gamma.requires_grad = True
                    self.g_a[i].gamma_reparam.pedestal = pretrained_model.g_a[i].gamma_reparam.pedestal
                    self.g_a[i].gamma_reparam.pedestal.requires_grad = True   
                else:
                    #print("implement initializer for windoe attention module")
                    self.g_a[i].initialize_weights(pretrained_model.g_a[i])
        #print("numbers of trainable parameters of the g_a: ", sum(p.numel() for p in self.g_a.parameters() if p.requires_grad))
        #print("numbers of FREEZED parameters of the g_a: ", sum(p.numel() for p in self.g_a.parameters() if not p.requires_grad)) 

        with torch.no_grad():
            for i,l in enumerate(pretrained_model.g_s):
                if i in [1,3,6,8]: # convolutional level 2d
                    self.g_s[i].weight = pretrained_model.g_s[i].weight
                    self.g_s[i].weight.requires_grad = True
                    self.g_s[i].bias = pretrained_model.g_s[i].bias
                    self.g_s[i].requires_grad = True
                elif i in [2,4,7]:
                    self.g_s[i].beta = pretrained_model.g_s[i].beta
                    self.g_s[i].beta.requires_grad=True
                    self.g_s[i].beta_reparam.pedestal = pretrained_model.g_s[i].beta_reparam.pedestal
                    self.g_s[i].beta_reparam.pedestal.requires_grad = True
                    self.g_s[i].gamma = pretrained_model.g_s[i].gamma
                    self.g_s[i].gamma.requires_grad = True
                    self.g_s[i].gamma_reparam.pedestal = pretrained_model.g_s[i].gamma_reparam.pedestal
                    self.g_s[i].gamma_reparam.pedestal.requires_grad = True   
                else:
                    #print("implement initializer for windoe attention module")
                    self.g_s[i].initialize_weights(pretrained_model.g_s[i])
        #print("numbers of trainable parameters of the g_s: ", sum(p.numel() for p in self.g_s.parameters() if p.requires_grad))
        #print("numbers of FREEZED parameters of the g_s: ", sum(p.numel() for p in self.g_s.parameters() if not p.requires_grad)) 

    def initialize_entropy_model(self, entropy_model = None):
        """
        Serve se voglio reinizializzare da capo il mio modello entropico!
        """
        if entropy_model is None:
            filters = (1,) + self.entropy_bottleneck.filters + (1,)
            scale = self.entropy_bottleneck.init_scale ** (1 / (len(self.entropy_bottleneck.filters) + 1))
            for i in range(len(self.entropy_bottleneck.filters) + 1):
                init = np.log(np.expm1(1 / scale / filters[i + 1]))
                #matrix = torch.Tensor(channels, filters[i + 1], filters[i])
                matrix = getattr(self.entropy_bottleneck, f"_matrix{i:d}")
                matrix.data.fill_(init)
                self.entropy_bottleneck.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))

                bias = getattr(self.entropy_bottleneck, f"_bias{i:d}")
                nn.init.uniform_(bias, -0.5, 0.5)
                self.entropy_bottleneck.register_parameter(f"_bias{i:d}", nn.Parameter(bias))

                if i < len(self.self.entropy_bottleneck.filters):
                    factor = getattr(self.entropy_bottleneck, f"_factor{i:d}")
                    nn.init.zeros_(factor)
                    self.entropy_bottleneck.register_parameter(f"_factor{i:d}", nn.Parameter(factor)) 
        else:
            #print("CHECK HERE!!! take baseline entropy model")
            num_params = 0
            for i in range(len(self.entropy_bottleneck.filters) + 1):
                matrix = getattr( self.entropy_bottleneck, f"_matrix{i:d}")
                num_params +=  matrix.reshape(-1).shape[0]
                self.entropy_bottleneck.register_parameter(f"_matrix{i:d}", nn.Parameter(getattr(entropy_model,f"_matrix{i:d}").data))
                bias = getattr( self.entropy_bottleneck, f"_bias{i:d}")
                num_params +=  bias.reshape(-1).shape[0]
                self.entropy_bottleneck.register_parameter(f"_bias{i:d}", nn.Parameter(getattr(entropy_model, f"_bias{i:d}").data))
                if i < len(self.entropy_bottleneck.filters):
                    factor = getattr(self.entropy_bottleneck, f"_factor{i:d}")
                    num_params +=  factor.reshape(-1).shape[0]
                    self.entropy_bottleneck.register_parameter(f"_factor{i:d}", nn.Parameter( getattr(entropy_model, f"_factor{i:d}").data)) 

            #print("number of countet factos: ",num_params)

    def initialize_hyperprior(self, pretrained_model): 
        # initialize h_a
        with torch.no_grad():
            for i,l in enumerate(pretrained_model.h_a):
                if i%2==0:
                    self.h_a[i].weight = pretrained_model.h_a[i].weight
                    self.h_a[i].weight.requires_grad = True
                    self.h_a[i].bias = pretrained_model.h_a[i].bias
                    self.h_a[i].requires_grad = True
            # initialize h_mean_s 
            for i,l in enumerate(pretrained_model.h_mean_s):
                if i in [0, 4, 8]:
                    self.h_mean_s[i].weight = pretrained_model.h_mean_s[i].weight
                    self.h_mean_s[i].weight.requires_grad = True
                    self.h_mean_s[i].bias = pretrained_model.h_mean_s[i].bias
                    self.h_mean_s[i].requires_grad = True

                    self.h_scale_s[i].weight = pretrained_model.h_scale_s[i].weight
                    self.h_scale_s[i].weight.requires_grad = True
                    self.h_scale_s[i].bias = pretrained_model.h_scale_s[i].bias
                    self.h_scale_s[i].requires_grad = True

                elif i in [2,6]:
                    for j, pp in enumerate(pretrained_model.h_mean_s[i]):
                        if j%2==0:
                            self.h_mean_s[i][j].weight = pretrained_model.h_mean_s[i][j].weight
                            self.h_mean_s[i][j].weight.requires_grad = True
                            self.h_mean_s[i][j].bias = pretrained_model.h_mean_s[i][j].bias
                            self.h_mean_s[i][j].requires_grad = True

                            self.h_scale_s[i][j].weight = pretrained_model.h_scale_s[i][j].weight
                            self.h_scale_s[i][j].weight.requires_grad = True
                            self.h_scale_s[i][j].bias = pretrained_model.h_scale_s[i][j].bias
                            self.h_scale_s[i][j].requires_grad = True
                else:
                    continue

    def initialize_cc(self, pretrained_model):

        with torch.no_grad():
            for j in range(10):
                for i,p in enumerate(self.cc_mean_transforms[j]):
                    if i%2==0:

                        self.cc_mean_transforms[j][i].weight = pretrained_model.cc_mean_transforms[j][i].weight
                        self.cc_mean_transforms[j][i].weight.requires_grad = True
                        self.cc_mean_transforms[j][i].bias = pretrained_model.cc_mean_transforms[j][i].bias
                        self.cc_mean_transforms[j][i].requires_grad = True                      

                        self.cc_scale_transforms[j][i].weight = pretrained_model.cc_scale_transforms[j][i].weight
                        self.cc_scale_transforms[j][i].weight.requires_grad = True
                        self.cc_scale_transforms[j][i].bias = pretrained_model.cc_scale_transforms[j][i].bias
                        self.cc_scale_transforms[j][i].requires_grad = True 

                        self.lrp_transforms[j][i].weight = pretrained_model.lrp_transforms[j][i].weight
                        self.lrp_transforms[j][i].weight.requires_grad = True
                        self.lrp_transforms[j][i].bias = pretrained_model.lrp_transforms[j][i].bias
                        self.lrp_transforms[j][i].requires_grad = True      

    
    def compress(self,x): 

        y = self.g_a(x)
        y_shape = y.shape[2:]
        #print("Y SHAPE------> ",y_shape)

        z = self.h_a(y)
    
        #al solito, tengo il vecchio modo perché non si sa mai
        z_strings, entropy_bottleneck_cdf = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress( z_strings, entropy_bottleneck_cdf)  

        



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
            print("slice index: ",slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            perm, inv_perm = self.define_permutation(y_slice)


            strings, cdfs, shapes_symb = self.gaussian_conditional.compress(y_slice, index,  [perm, inv_perm], means = mu) # shape is flattenend ( in theory)


            y_q_slice = self.gaussian_conditional.quantize(y_slice, mode = "symbols", means = mu, perms = [perm,inv_perm]) #questo va codificato!!!
            proper_shape = y_q_slice.shape

            
            y_strings.append(strings) 
            y_cdfs.append(cdfs)
            y_shapes.append(proper_shape)

            y_q_slice = self.gaussian_conditional.dequantize(y_q_slice) 
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





    def decompress(self,data):
        strings = data["strings"] 
        cdfs = data["cdfs"]
        shapes = data["shapes"]
        z_hat = self.entropy_bottleneck.decompress(strings[0],cdfs[0])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[1]
        y_cdf = cdfs[1]

        y_hat_slices = []

        for slice_index in range(self.num_slices):
            print("slice index: ",slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            #index = self.gaussian_conditional.build_indexes(scale)




            rv = self.gaussian_conditional.decompress(y_string[slice_index],y_cdf[slice_index]) # decompress -> dequantize  + mu
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
    
        
    



