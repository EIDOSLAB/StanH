import torch 
import os 
import numpy as np 
from pathlib import Path
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import math
from compressai.ops import compute_padding
import math 
from pytorch_msssim import ms_ssim
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
from compressai.zoo import *
from torch.utils.data import DataLoader
from os.path import join 
from compress.zoo import models, aux_net_models
import wandb
from typing import  Dict, NamedTuple, Tuple
import pickle
import psutil


def upload_model(models_dict, qual, baseline,  device):
    # models_dict["net"] contiene lo state_dict per caricare il modello
    if baseline is True: 
        state_dict = load_pretrained(models_dict[qual]['net'])
        model = from_state_dict(aux_net_models["cnn"], state_dict) #.eval()
        model.update()
        model.to(device)  
    else:
        if qual in ["q8","q5","q2"]:
            anch_qual = qual
        elif qual == "q7":
            anch_qual = "q8"
        elif qual == "q3":
            anch_qual = "q5"
        else:
            anch_qual = "q2"

        model = models["cnn"](192, 320, factorized_configuration = models_dict[anch_qual]["factorized_configuration"], gaussian_configuration = models_dict[anch_qual]["gaussian_configuration"])
        model = model.to(device)
        model.load_state_dict(models_dict[anch_qual]["state_dict"])                      
        model.entropy_bottleneck.sos.update_state(device = device )
        model.gaussian_conditional.sos.update_state(device = device)
        model.update( device = device)

        if qual != anch_qual: #siamo in una derivazione
            model.entropy_bottleneck.sos.w = models_dict[anch_qual]["stanh_der"][0]
            model.entropy_bottleneck.sos.w = models_dict[anch_qual]["stanh_der"][1]

            model.gaussian_conditional.sos.w = models_dict[anch_qual]["stanh_der"][2]
            model.gaussian_conditional.sos.w = models_dict[anch_qual]["stanh_der"][3]

            model.entropy_bottleneck.sos.update_state(device = device )
            model.gaussian_conditional.sos.update_state(device = device)

        
    return model




            

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



def from_state_dict(cls, state_dict):
    net = cls()
    net.load_state_dict(state_dict)
    return net


def load_pretrained(state_dict):
    """Convert sccctaddte_dict keys."""
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    if None in state_dict:
        state_dict.pop(None)
    return state_dict

class CodecInfo(NamedTuple):
    codec_header: Tuple
    original_size: Tuple
    original_bitdepth: int
    net: Dict
    device: str


def save_models(path, savepath, device = torch.device("cpu"), baseline = True ):
    """
    funzione per salvare i modelli da utilizzare. 
    baseline = {
    
    "quality_1":{ "net": net
                }
     .... 

     "quality_6: 
    }

    ours = {
        "quality_1 : {"net": net , "stanh1": [], "stanh2": []}
        ... "quality_3
    }
    """

    models_list = os.listdir(path)
    res = {}
    if baseline:
        print("entro qua")
        for ml in models_list: 
            
            qual = ml.split("-")[0] #q1, q2 ....
            complete_path = os.path.join(path, ml) 
            res[qual] = {}
            pt = os.path.join(complete_path)              
            state_dict = load_pretrained(torch.load(pt, map_location=device)['state_dict'])
            model = from_state_dict(aux_net_models["cnn"], state_dict) #.eval()
            model.update()
            model.to(device) 
            res[qual] = {"state_dict": state_dict}
            print(ml,"  ",qual)
    else: 
        anchors = ["q8","q5","q2"]
        dict_derivates = {"q8":"q7","q5":"q3","q2":"q1"}
        for an in anchors: 
            for ml in models_list:  
                if an in ml: 
                    print("anchors<<<<<<: ",an)
                    # salvo modello ancora 
                    complete_path = os.path.join(path, ml)
                    checkpoint = torch.load(complete_path, map_location=device)
  

                    model = models["cnn"](192, 320, factorized_configuration = checkpoint["factorized_configuration"], gaussian_configuration = checkpoint["gaussian_configuration"])
                    model = model.to(device)
                    


                    state_dict = checkpoint["state_dict"]

                    model.load_state_dict(checkpoint["state_dict"]) 
                     
                    model.entropy_bottleneck.sos.update_state(device = device )
                    model.gaussian_conditional.sos.update_state(device = device)

                    model.update( device = device)

                    # ora considero solamente 
                    ders = dict_derivates[an]

                    name_model = ders + "-zou22.pth.tar"
                    complete_path = os.path.join(path, name_model)
                    checkpoint_der = torch.load(complete_path, map_location=device)


                    der = models["cnn"](192, 320, factorized_configuration = checkpoint_der["factorized_configuration"], gaussian_configuration = checkpoint_der["gaussian_configuration"])
                    
                    der.update( device = device)
                    der.load_state_dict(checkpoint_der["state_dict"])  
                    der.entropy_bottleneck.sos.update_state(device = device )
                    der.gaussian_conditional.sos.update_state(device = device)  
                    der = der.to(device)



                    res[an] = {"state_dict": state_dict, 
                                                  "stanh_der": [der.entropy_bottleneck.sos.w, der.entropy_bottleneck.sos.b , der.gaussian_conditional.sos.w, der.gaussian_conditional.sos.b],
                                                  "factorized_configuration": checkpoint["factorized_configuration"], 
                                                  "gaussian_configuration":checkpoint["gaussian_configuration"]}


    pth = os.path.join(savepath, "_models_dict.pkl")
    print(res.keys())
    # save dictionary to person_data.pkl file
    with open(pth, 'wb') as fp:
        pickle.dump(res, fp)
        print('dictionary saved successfully to file')
 
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("-m","--model",default="3anchorsbis",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-mp","--model_path",default="/scratch/inference/new_models/devil2022",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-mdp","--model_dict_path",default="/scratch/inference/bitstream/devil2022",help="Model architecture (default: %(default)s)",)

    
    parser.add_argument("-sp","--savepath",default="/scratch/inference/bitstream/devil2022/",help="Model architecture (default: %(default)s)",)

    parser.add_argument("-ip","--image_path",default="/scratch/dataset/kodak/kodim21.png",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-bl","--baseline",default=True)
    parser.add_argument("-ql","--qual",default="q5",help="Model architecture (default: %(default)s)",)


    args = parser.parse_args(argv)
    return args

@torch.no_grad()
def encode_bitstream(model, files, device, quality, savepath):
    print("inizio inferenza!")

    x = read_image(files).to(device)
    x = x.unsqueeze(0) 
    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    x_padded = F.pad(x, pad, mode="constant", value=0)


    data =  model.compress(x_padded)

    pth = os.path.join(savepath,str(quality) + "_files.pkl")
    print("-------> ",pth)
    with open(pth, 'wb') as fp:
        pickle.dump(data, fp)
        print('dictionary saved successfully to file')      
 


def read_image(filepath):
    #assert filepath.is_file()
    img = Image.open(filepath)
    img = img.convert("RGB")
    return transforms.ToTensor()(img)






def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def main(argv):
    set_seed()
    args = parse_args(argv)
    #model_name = args.model  
    #models_path = os.path.join(args.model_path,model_name)
    #savepath = args.savepath 
    #save_models(models_path, savepath, device = torch.device("cpu"), baseline = args.baseline )

    device = torch.device("cpu")
    
    model_dict_path = args.model_dict_path
    if args.baseline is True:
        pth_complete = os.path.join(model_dict_path,"baseline_decoder","_models_dict.pkl")
        pth_save = os.path.join(args.savepath,"baseline_stream")
    else:
        pth_complete = os.path.join(model_dict_path,"ours_decoder","_models_dict.pkl")
        pth_save = os.path.join(args.savepath,"ours_stream")

    files = args.image_path

    # Ottieni le informazioni sulla memoria
    memory_info = psutil.virtual_memory()

    used_memory_1 = memory_info.used

    with open(pth_complete, 'rb') as fp:
        print(pth_complete,"<----------------questo è il pth completo")
        models = pickle.load(fp)
        print('uploaded dictionary')

    memory_info = psutil.virtual_memory()

    # Ottieni la quantità di memoria utilizzata in byte
    used_memory_2 = memory_info.used
    # Converti la memoria utilizzata in unità più comuni come MB o GB
    used_memory_mb = (used_memory_2 -used_memory_1)  / (1024 * 1024)
    used_memory_gb = (used_memory_2 -used_memory_1) / (1024 * 1024 * 1024)
    print(f"Memoria utilizzata: {used_memory_2 - used_memory_1} byte")
    print(f"Memoria utilizzata: {used_memory_mb} MB")
    print(f"Memoria utilizzata: {used_memory_gb} GB")


    #if args.baseline: 
    #    lista_quality = ["q6","q5","q4","q3","q2","q1"]#list(models.keys())
    #else:
    #    lista_quality = ["q8","q7","q5","q3","q2","q1"]

    lista_quality = [args.qual]

    for q in lista_quality:  
        print("il baseline è ",args.baseline)
        net = upload_model(models, q, args.baseline, device)

        encode_bitstream(net, files, device, q, pth_save)
        print(q," done")
        print("EVERYTHING HAS BEEN DONE")


    





if __name__ == "__main__":

    wandb.init(project="encoded_bitream", entity="albertopresta")   

    main(sys.argv[1:])


