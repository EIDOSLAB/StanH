import torch.nn as nn
import torch
import shutil
import torch.optim as optim
import math
from pytorch_msssim import ms_ssim
import torch.nn.functional as F
import torchac

def configure_latent_space_policy(args):


    if args.fact_num_sigmoids == -1:
        factorized_configuration = None 
    else:
        if args.fact_tr is not None:
            factorized_configuration = {
                "beta": args.fact_beta, 
                "num_sigmoids": args.fact_num_sigmoids, 
                "activation": args.fact_activation, 
                "annealing": args.fact_annealing, 
                "gap_factor": args.fact_gp,
                "extrema": args.fact_extrema,
              "trainable":args.fact_tr
            }
        else:
            factorized_configuration = {
                "beta": args.fact_beta, 
                "num_sigmoids": args.fact_num_sigmoids, 
                "activation": args.fact_activation, 
                "annealing": args.fact_annealing, 
                "gap_factor": args.fact_gp,
                "extrema": args.fact_extrema,

            }
          
    if args.gauss_num_sigmoids == -1:
        gaussian_configuration = None 
    else:
        if args.gauss_tr is not None:
            gaussian_configuration = {
                "beta": args.gauss_beta, 
                "num_sigmoids": args.gauss_num_sigmoids, 
                "activation": args.gauss_activation, 
                "annealing": args.gauss_annealing, 
                "gap_factor": args.gauss_gp ,
                "extrema": args.gauss_extrema ,
            "trainable":args.gauss_tr         
            }
        else:
            print("dovrei entrare qua dal momento che non ho configuration")
            gaussian_configuration = {
                "beta": args.gauss_beta, 
                "num_sigmoids": args.gauss_num_sigmoids, 
                "activation": args.gauss_activation, 
                "annealing": args.gauss_annealing, 
                "gap_factor": args.gauss_gp ,
                "extrema": args.gauss_extrema ,
     
            }



    print("factorized configuration: ",factorized_configuration)
    return factorized_configuration, gaussian_configuration

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])




from datetime import datetime
from os.path import join         
def create_savepath(args,epoch):
    now = datetime.now()
    date_time = now.strftime("%m%d")
    c = join(date_time,"_lambda_",str(args.lmbda),"_epoch_",str(epoch)).replace("/","_")

    
    c_best = join(c,"best").replace("/","_")
    c = join(c,args.suffix).replace("/","_")
    c_best = join(c_best,args.suffix).replace("/","_")
    
    
    path = args.filename
    savepath = join(path,c)
    savepath_best = join(path,c_best)
    
    print("savepath: ",savepath)
    print("savepath best: ",savepath_best)
    return savepath, savepath_best





def load_pretrained_net_for_training( mod_load,  architecture,   ml):

    N = mod_load["N"]
    M = mod_load["M"]


    print("N and M is: ",N,"  ",M)
    
    factorized_configuration = mod_load["factorized_configuration"]
    gaussian_configuration =  mod_load["gaussian_configuration"]




    factorized_configuration["trainable"] = True
    gaussian_configuration["trainable"] = True



    #models[args.model](N = N, M = M, factorized_configuration = factorized_configuration, gaussian_configuration = gaussian_configuration , baseline = args.baseline, pretrained_model = aux_net)
    model = architecture(N = N,M = M,factorized_configuration = factorized_configuration, gaussian_configuration = gaussian_configuration)  
    model = model.to(ml)
    print("performing the state dict")
    #model.update()
    del mod_load["state_dict"]["gaussian_conditional._offset"] 
    del mod_load["state_dict"]["gaussian_conditional._quantized_cdf"] 
    del mod_load["state_dict"]["gaussian_conditional._cdf_length"] 
    del mod_load["state_dict"]["gaussian_conditional.scale_table"] 

    model.load_state_dict(mod_load["state_dict"])  
    model.update()
    model.entropy_bottleneck.sos.update_state()
    model.gaussian_conditional.sos.update_state()
    print("***********************************************************************************************************")
    print("cumulative weights update222d----> ",model.entropy_bottleneck.sos.cum_w)
    #model.entropy_bottleneck.sos.update_cumulative_weights()    
    model.update( device = torch.device("cuda"))

    annealing_strategy_bottleneck =  mod_load["annealing_strategy_bottleneck"]
    annealing_strategy_gaussian = mod_load["annealing_strategy_gaussian"]
    return model, annealing_strategy_bottleneck, annealing_strategy_gaussian



def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())

    if args.baseline:

        aux_parameters = {
            n
            for n, p in net.named_parameters()
            if n.endswith(".quantiles") and p.requires_grad
        }

        # Make sure we don't have an intersection of parameters
        params_dict = dict(net.named_parameters())
        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters

        assert len(inter_params) == 0
        assert len(union_params) - len(params_dict.keys()) == 0


        aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)),lr=args.aux_learning_rate,)
    else:
        aux_optimizer = None

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    return optimizer, aux_optimizer
import wandb
def save_checkpoint_our(state, is_best, filename,filename_best):
    torch.save(state, filename)
    wandb.save(filename)
    if is_best:
        shutil.copyfile(filename, filename_best)
        wandb.save(filename_best)



def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-8]+"_best"+filename[-8:])



##########################################################################################
#########################################################################################
##########################################################################################

def pmf_to_cdf(pmf):
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    return cdf_with_0


def compute_cdf_for_baselines(net):
    scale_table = get_scale_table()
    net.gaussian_conditional.update_scale_table(scale_table)
    multiplier = -net.gaussian_conditional._standardized_quantile(net.gaussian_conditional.tail_mass / 2)
    pmf_center = torch.ceil(net.gaussian_conditional.scale_table * multiplier).int()
    pmf_length = 2 * pmf_center + 1
    max_length = torch.max(pmf_length).item()

    device = pmf_center.device
    samples = torch.abs(torch.arange(max_length, device=device).int() - pmf_center[:, None])
    samples_scale = net.gaussian_conditional.scale_table.unsqueeze(1)
    samples = samples.float()
    samples_scale = samples_scale.float()
    upper = net.gaussian_conditional._standardized_cumulative((0.5 - samples) / samples_scale)
    lower = net.gaussian_conditional._standardized_cumulative((-0.5 - samples) / samples_scale)
    pmf = upper - lower
    net.gaussian_conditional.pmf = pmf
    #tail_mass = 2 * lower[:, :1]
    net.gaussian_conditional.cdf =  pmf_to_cdf(pmf)
    net.cdf = net.gaussian_conditional.cdf 
    #quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
    #quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
    #self._quantized_cdf = quantized_cdf
    #self._offset = -pmf_center
    #self._cdf_length = pmf_length + 2
import time
def compress_with_torchac(net, x):


    y = net.g_a(x)  # [1,192,16,16]
    z = net.h_a(y) # [1,192,4,4]


    
    z_strings = net.entropy_bottleneck.compress(z)
    start_int = time.time()
    z_hat = net.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
    print("internal time /check if it better: ",time.time() - start_int)
    params = net.h_s(z_hat)

    s = 4  # scaling factor between z and y  
    kernel_size = 5  # context prediction kernel size
    padding = (kernel_size - 1) // 2  #2
    y_height = z_hat.size(2) * s #16
    y_width = z_hat.size(3) * s #16

    y_hat = F.pad(y, (padding, padding, padding, padding)) #[1,192,20,20]

    i = 0
    string, cdf, maps = _compress_ar(net, y_hat[i : i + 1],params[i : i + 1],y_height,y_width,kernel_size,padding,)
    y_strings = string #.append(string)
    cdfs = cdf #append(cdf)      

    #print("qua è finito tutto, ma no cicli")
    #print(z.size()[-2:],"<--- questo è l'unico shape che ci portiamo dietro")
    return {"strings": [z_strings, y_strings], "shape": [z.size()[-2:]], "cdfs": ["none", cdfs], "maps":maps}



def apply_map(y_q, map, inv = False):
    shp = y_q.shape
    y_q = y_q.ravel()
    for i in range(y_q.shape[0]):

        y_q[i] = map[y_q[i].item()]
    y_q = y_q.reshape(shp)
    return y_q

def finds_map(y_q):

    unique_values = torch.unique(y_q)
    c = 0
    map = {}
    inv_map = {}
    for i,uv in enumerate(unique_values):
        map[uv.item()] = i
        inv_map[i] = uv.item()

    return map, inv_map

def _compress_ar(net, y_hat, params, height, width, kernel_size, padding):
    symbols_list = []
    cdf_list = []
    maps = []
    masked_weight = net.context_prediction.weight * net.context_prediction.mask
    for h in range(height):
        for w in range(width):
            y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size] #[1,192,5,5]
            ctx_p = F.conv2d(y_crop,masked_weight,bias=net.context_prediction.bias,)
            p = params[:, :, h : h + 1, w : w + 1]
            gaussian_params = net.entropy_parameters(torch.cat((p, ctx_p), dim=1))
            gaussian_params = gaussian_params.squeeze(3).squeeze(2)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)  #[1,192], [1,192]

            indexes = net.gaussian_conditional.build_indexes(scales_hat)
            y_crop = y_crop[:, :, padding, padding]


            y_q = net.gaussian_conditional.quantize(y_crop,"symbols", means = means_hat)

            map, inv_map = finds_map(y_q)
            #minimo = torch.min(y_q).item()
            y_q = apply_map(y_q, map)
            #y_q = y_q - minimo
            y_sym, gaussian_cdf , _  = compress_with_torch(net, y_q, indexes)
            y_q = apply_map(y_q, inv_map)
            #y_q = y_q + minimo
            y_hat[:, :, h + padding, w + padding] = y_q  + means_hat

            symbols_list.append(y_sym)

            maps.append(inv_map)
            #maps.append(minimo)
            cdf_list.append(gaussian_cdf)



    return symbols_list, cdf_list, maps


def decompress_with_torchac(net, data):


    strings = data["strings"]
    cdfs = data["cdfs"]
    inv_map = data["maps"]
    shapes = data["shape"][0]

    z_hat = net.entropy_bottleneck.decompress(strings[0], shapes)
    params = net.h_s(z_hat)
    #print("in decoding: ",z_hat.shape)


    s = 4  # scaling factor between z and y
    kernel_size = 5  # context prediction kernel size
    padding = (kernel_size - 1) // 2

    y_height = z_hat.size(2) * s #16
    y_width = z_hat.size(3) * s #16

    y_hat = torch.zeros((z_hat.size(0), net.M, y_height + 2 * padding, y_width + 2 * padding),device=z_hat.device,)


    print("lunghezza in decoding !: ",len(strings[1]),"   ",len(cdfs[1]))
    i = 0

    _decompress_ar(net,strings[1],y_hat[i : i + 1],params[i : i + 1],y_height,y_width,kernel_size,padding,cdfs[1],  inv_map)

    y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
    x_hat = net.g_s(y_hat).clamp_(0, 1)
    return {"x_hat": x_hat}

def _decompress_ar(net, y_string, y_hat, params, height, width, kernel_size, padding, gaussian_cdf,  inv_map):

    cont = 0
    for h in range(height):
        for w in range(width):
            # only perform the 5x5 convolution on a cropped tensor
            # centered in (h, w)
            y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
            ctx_p = F.conv2d(y_crop,net.context_prediction.weight,bias=net.context_prediction.bias,)

            p = params[:, :, h : h + 1, w : w + 1]
            gaussian_params = net.entropy_parameters(torch.cat((p, ctx_p), dim=1))
            scales_hat, means_hat = gaussian_params.chunk(2, 1)

            indexes = net.gaussian_conditional.build_indexes(scales_hat)
            rv = decompress_with_torch(y_string[cont], gaussian_cdf[cont])
            cdf_tr = net.gaussian_conditional.retrieve_cdf_from_indexes(rv.shape[0], indexes.ravel())
            rv = apply_map(rv, inv_map[cont])
            #rv = rv + inv_map[cont] # ci aggiungo il minimo per mappare indietro il tutto (stessa cosa dell'offset)
            cont += 1 

            

            rv = rv.reshape(1, -1, 1, 1)
            rv = rv.to(means_hat.device)
            #print("lo shape di rv DOPO è il seguente!: ",rv.shape,"    ",means_hat.shape)
            rv = rv + means_hat # self.gaussian_conditional.dequantize(rv, means_hat)

            hp = h + padding
            wp = w + padding
            y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv




def  compress_with_torch(net, inputs, indexes):


    symbols = inputs #[1,128,32,48]
    shape_symbols = symbols.shape


    symbols = symbols.ravel().to(torch.int16)
    indexes = indexes.ravel().to(torch.int16)

        
    symbols = symbols.to("cpu")  

    output_cdf = torch.zeros_like(symbols)
    output_cdf = output_cdf[:,None] + torch.zeros(net.cdf.shape[1])
    output_cdf = output_cdf.to("cpu")
    for i in range(symbols.shape[0]):
        output_cdf[i,:] = net.cdf[indexes[i].item(),:]
    byte_stream = torchac.encode_float_cdf(output_cdf, symbols, check_input_bounds=True)

    c = torchac.decode_float_cdf(output_cdf, byte_stream)

    if torchac.decode_float_cdf(output_cdf, byte_stream).equal(symbols) is False:
        raise ValueError("L'output Gaussiano codificato è !diverso, qualcosa non va!")
        #else:
            #print("l'immagine è ok, non ci sono problemi!")
    return byte_stream, output_cdf  , shape_symbols



def decompress_with_torch( byte_stream, output_cdf):
    output = torchac.decode_float_cdf(output_cdf, byte_stream)#.type(torch.FloatTensor)
    return output

import math
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))