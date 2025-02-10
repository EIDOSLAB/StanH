
import torch
from compress.zoo import models, aux_net_models
import shutil 
import wandb
from datetime import datetime
from os.path import join
from collections import OrderedDict

def save_checkpoint_our(state, is_best, filename,filename_best):
    torch.save(state, filename)
    wandb.save(filename)
    if is_best:
        shutil.copyfile(filename, filename_best)
        wandb.save(filename_best)



def create_savepath(args):
    now = datetime.now()
    date_time = now.strftime("%m%d")
    suffix = ".pth.tar"
    c = join(date_time,"last").replace("/","_")

    
    c_best = join(date_time,"best").replace("/","_")
    c = join(c,suffix).replace("/","_")
    c_best = join(c_best,suffix).replace("/","_")
    
    
    path = args.filename
    savepath = join(path,c)
    savepath_best = join(path,c_best)
    
    print("savepath: ",savepath)
    print("savepath best: ",savepath_best)
    return savepath, savepath_best









def InsertStanHOnCheckpoints(state_dict,num_stanh):

    res =  OrderedDict()


    for k,v in state_dict.items():
        if "gaussian_conditional" in k:
            for j in range(num_stanh):
                adding = str(j) 
                new_text = k.replace("gaussian_conditional.", "gaussian_conditional." + adding + ".")
                res[new_text] = state_dict[k]
        else:
            res[k]=state_dict[k]
    
    return res



image_models = {"zou22-base": aux_net_models["stf"],
                "zou22-sos":models["cnn_multi"],

                }

def rename_key(key):
    """Rename state_deeict key."""

    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]
    if key.startswith('h_s.'):
        return None

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    # if ".downsample." in key:
    #     return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key



def load_state_dict(state_dict):
    """Convert state_dict keys."""
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    if None in state_dict:
        state_dict.pop(None)
    return state_dict

def load_checkpoint(arch: str, checkpoint_path: str):
    state_dict = load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return models[arch].from_state_dict(state_dict).eval()

