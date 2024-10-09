


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
from torch.utils.data import Dataset
from os import listdir

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

image_models = {"zou22-base": aux_net_models["stf"],
                "zou22-sos":models["cnn"]
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



@torch.no_grad()
def test_epoch( test_dataloader, model,  sos):
    model.eval()
    device = next(model.parameters()).device
    bpp_loss = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()
    with torch.no_grad():
        for i,d in enumerate(test_dataloader):
            d = d.to(device)
            if sos:
                out_net = model(d, training = False)
            else: 
                out_net = model(d)


            N, _, H, W = out_net["x_hat"].size() 
            num_pixels = N*W*H
            bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_net["likelihoods"].values())
            bpp_loss.update(bpp)
            psnr.update(compute_psnr(d, out_net["x_hat"]))
            ssim.update(compute_msssim(d, out_net["x_hat"]))
            print("IMMAGINE ",i,"_ ",bpp,"-",compute_psnr(d, out_net["x_hat"]),"-",compute_msssim(d, out_net["x_hat"]))





    return  psnr.avg, ssim.avg, bpp_loss.avg




class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = [os.path.join(self.data_dir,f) for f in os.listdir(self.data_dir)]

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose(
        [transforms.CenterCrop(256), transforms.ToTensor()]
    )
        return transform(image)

    def __len__(self):
        return len(self.image_path)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("-m","--model",default="4anchors",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-mp","--model_path",default="/scratch/inference/new_models/devil2022/",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-rp","--result_path",default="/scratch/inference/results",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-ip","--image_path",default="/scratch/dataset/kodak",help="Model architecture (default: %(default)s)",)
    parser.add_argument("--entropy_estimation", action="store_true", help="Use cuda")


    args = parser.parse_args(argv)
    return args

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)




def reconstruct_image_with_nn(networks, filepath, device, save_path):
    reconstruction = {}
    for name, net in networks.items():
        #net.eval()
        with torch.no_grad():
            x = read_image(filepath).to(device)
            x = x.unsqueeze(0)
            out_net,= net(x,  False)
            out_net["x_hat"].clamp_(0.,1.)
            original_image = transforms.ToPILImage()(x.squeeze())
            reconstruction[name] = transforms.ToPILImage()(out_net['x_hat'].squeeze())




    svpt = os.path.joint(save_path,"original" + filepath.split("/")[-1])

    fix, axes = plt.subplots(1, 1)
    for ax in axes.ravel():
        ax.axis("off")

    axes.ravel()[0 ].imshow(original_image)
    axes.ravel()[0].title.set_text("original image")    
    
    plt.savefig(svpt)
    plt.close()


    svpt = os.path.joint(save_path,filepath.split("/")[-1])

    fix, axes = plt.subplots(5, 4, figsize=(10, 10))
    for ax in axes.ravel():
        ax.axis("off")
    

    for i, (name, rec) in enumerate(reconstruction.items()):
            #axes.ravel()[i + 1 ].imshow(rec.crop((468, 212, 768, 512))) # cropped for easy comparison
        axes.ravel()[i ].imshow(rec)
        axes.ravel()[i].title.set_text(name)

        #plt.show()
    plt.savefig(svpt)
    plt.close()


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics( org, rec, max_val: int = 255):
    metrics =  {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics

def load_models(dict_model_list,  models_path, device, image_models ,desired_base_quality = [1,2,3,4,5,6]):

    res = {}
    for i, name in enumerate(list(dict_model_list.keys())):#dict_model_listload

        #"q10" in name or "a10" in name or q8
        print("*************** ",name)
        nm = "zou22-sos" #"cnn"name[4:].split(".")[0] if int(name.split("-")[0][1:]) >= 10 else name[3:].split(".")[0] 

        #nm_sos = nm + "-sos"
        #nm_base = nm + "-base"
        print("vedo che tipo di nome è venuto  ",nm,"<-------- ")
        checkpoint =  dict_model_list[name] 
        #if "sos" in nm:

            
        architecture =  image_models[nm]
        pt = os.path.join(models_path, checkpoint)
        checkpoint = torch.load(pt, map_location=device)


        factorized_configuration =[checkpoint["factorized_configuration"]]
        factorized_configuration[0]["trainable"] = True
        gaussian_configuration =  [checkpoint["gaussian_configuration"]]
        gaussian_configuration[0]["trainable"] = True
        model =architecture(N = 192, 
                            M = 320, 
                            factorized_configuration = factorized_configuration, 
                            gaussian_configuration = gaussian_configuration)
            
        model = model.to(device)
                          
        model.update( device = device)

        print("----> ",checkpoint["state_dict"].keys())
        checkpoint["state_dict"]["gaussian_conditional._cdf_length"] = checkpoint["state_dict"]["gaussian_conditional._cdf_length"].ravel()
        model.load_state_dict(checkpoint["state_dict"])  
    
        model.entropy_bottleneck.sos.update_state(device = device )
        model.gaussian_conditional.sos.update_state(device = device)
    
            


        model.update( device = device)         
        #torch.save({"state_dict": model.state_dict()},"/scratch/inference/baseline_models/zou2022/q1_1905.pth.tar")
        res[name] = { "model": model}
        print("------------------- ",name)
        print(model.gaussian_conditional.sos.w.shape)

        print("save stanh in a separate file")
        state_dict_stanh = {}
        state_dict_stanh["state_dict"] = {}
        state_dict_stanh["state_dict"]["gaussian_conditional"] = {}
        state_dict_stanh["state_dict"]["entropy_bottleneck"] = {}

        state_dict_stanh["state_dict"]["gaussian_conditional"]["w"] = checkpoint["state_dict"]["gaussian_conditional.sos.w"]
        state_dict_stanh["state_dict"]["gaussian_conditional"]["b"] = checkpoint["state_dict"]["gaussian_conditional.sos.b"]
        state_dict_stanh["state_dict"]["entropy_bottleneck"]["w"] = checkpoint["state_dict"]["entropy_bottleneck.sos.w"]
        state_dict_stanh["state_dict"]["entropy_bottleneck"]["b"] = checkpoint["state_dict"]["entropy_bottleneck.sos.b"]

        state_dict_stanh["factorized_configuration"] = checkpoint["factorized_configuration"]
        state_dict_stanh["gaussian_configuration"] = checkpoint["gaussian_configuration"]

        
        filename = "/scratch/inference/new_models/devil2022/3_anchors_stanh/" +  name.split("/")[-1].split("-")[0] + "-stanh.pth.tar"

        torch.save(state_dict_stanh, filename)


    # carico i modelli base
    #base_path = "/scratch/pretrained_models/zou22"
    #base_models = os.listdir(base_path)

    #for bm in base_models:
            
    #    qual = int(bm.split("-")[0][1:])
    #    if qual in desired_base_quality:

    #        pt = os.path.join("/scratch/pretrained_models/zou2022",bm)
    #        state_dict = load_pretrained(torch.load(pt, map_location=device))
    #        model = from_state_dict(aux_net_models["cnn"], state_dict) #.eval()
    #        model.update()
    #        model.to(device) 

    #        name = bm.split(".")[0] + "-base"
    #        res[name] = { "model": model}



            
    
    return res


def modify_dictionary(check):
    res = {}
    ks = list(check.keys())
    for key in ks: 
        res[key[7:]] = check[key]
    return res


def collect_images(rootpath: str):
    image_files = []

    for ext in IMG_EXTENSIONS:
        image_files.extend(Path(rootpath).rglob(f"*{ext}"))
    return sorted(image_files)


def read_image(filepath, clic =False):
    #assert filepath.is_file()
    img = Image.open(filepath)
    
    if clic:
        i =  img.size
        i = i[0]//2, i[1]//2
        img = img.resize(i)
    img = img.convert("RGB")
    return transforms.ToTensor()(img)





transl_sos = {  


            "q17":"17",
            "q21":"21",
            "q22":"22",
            "q23":"23",
            "q24":"24",
            "q25":"25",
            "q26":"26",
            "q31":"31",
            "q32":"32",
            "a10":"10",
            "a30":"30",
            "a60":"60"
              }

transl= {  
            "q6": "18",
              "q5": "22",
              "q4": "27",
              "q3":"32",
              "q2":"37",
              "q1":"42",
              "q7": "15"
              }







@torch.no_grad()
def inference(model, filelist, device, sos,model_name, entropy_estimation = False):
    # tolgo il dataloader al momento
    psnr = AverageMeter()
    ms_ssim = AverageMeter()
    bpps = AverageMeter()
    quality_level =model_name.split("-")[0]
    print("inizio inferenza -----> ",model_name)
    i = 0
    for d in filelist:
        name = "image_" + str(i)
        print(name," ",d," ",i)
        i +=1
        x = read_image(d).to(device)
        x = x.unsqueeze(0) 
        h, w = x.size(2), x.size(3)
        pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
        x_padded = F.pad(x, pad, mode="constant", value=0)
        imgg = transforms.ToPILImage()(x_padded.squeeze())
        
        #d = d.to("cpu")
        #x_padded = d 
        #unpad = 0
        #imgg = transforms.ToPILImage()(d)
        #print("lo shape at encoded is: ",d.shape)
        #data =  model.compress(x_padded)
        if entropy_estimation is False:
            #print("entro qua!!!!")
            data =  model.compress(x_padded)
            if sos: 
                out_dec = model.decompress(data)
            else:
                out_dec = model.decompress(data["strings"], data["shape"])
        else:
            if sos: 
                out_dec = model(x_padded, training = False)
            else: 
                out_dec = model(x_padded)
        if entropy_estimation is False:
            out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
            out_dec["x_hat"].clamp_(0.,1.)
            metrics = compute_metrics(x, out_dec["x_hat"], 255)
            size = out_dec['x_hat'].size()
            num_pixels = size[0] * size[2] * size[3]
            if sos:
                bpp ,_, _= bpp_calculation(out_dec, data["strings"])
            else:
                bpp = sum(len(s[0]) for s in data["strings"]) * 8.0 / num_pixels
            
            metrics = compute_metrics(x_padded, out_dec["x_hat"], 255)
            print("fine immagine: ",bpp," ",metrics)

        else:
            out_dec["x_hat"].clamp_(0.,1.)
            out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
            size = out_dec['x_hat'].size()
            num_pixels = size[0] * size[2] * size[3]
            bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_dec["likelihoods"].values())
            metrics = compute_metrics(x, out_dec["x_hat"], 255)
            print("fine immagine: ",bpp," ",metrics)
        
        
        if i <= -25:
            if sos is False:

                folder_path = "/scratch/inference/images/devil2022/base/" + transl_sos[quality_level] 
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    print(f"Cartella '{folder_path}' creata.")

                image = transforms.ToPILImage()(out_dec['x_hat'].squeeze())
                nome_salv = os.path.join(folder_path, name + ".png")
                image.save(nome_salv)

                #nome_salv2 = "/scratch/inference/results/images/devil2022/" + name + "original" +  ".png"
                #imgg.save(nome_salv2)
            else:
                print("else")                #folder_path = "/scratch/inference/images/devil2022/6anchors/" + transl_sos[quality_level] 
                #if not os.path.exists(folder_path):
                #    os.makedirs(folder_path)
                #    print(f"Cartella '{folder_path}' creata.")
                #else:
                #    print(f"La cartella '{folder_path}' esiste già.")
                #image = transforms.ToPILImage()(out_dec['x_hat'].squeeze())
                #nome_salv = os.path.join(folder_path, name + ".png")#"/scratch/inference/images/devil2022/3anchors" + name +    ".png"
                #image.save(nome_salv)

        

        psnr.update(metrics["psnr"])
        if i%8==1:
            print(name,": ",metrics["psnr"]," ",bpp, metrics["ms-ssim"])
        ms_ssim.update(metrics["ms-ssim"])
        #bpps.update(bpp.item())
        bpps.update(bpp)

        modality = "None"
        """
        if sos:
            modality = "prop"
            f=open("/scratch/inference/results/clic/bjonte/devil2022/sos_clic_2606_FINALE_6ANCHORS.txt" , "a+")
            f.write("MODE " + modality + " SEQUENCE " + name +  " QP " +  transl_sos[quality_level] + " BITS " +  str(bpp) + " YPSNR " +  str(metrics["psnr"])  + " YMSSIM " +  str(metrics["ms-ssim"]) + "\n")

        else:
            modality = "ref"
            f=open("/scratch/inference/results/tecnik/bjonte/devil2022/devil2022_tecnik_1906_FINALE_BASELINE.txt" , "a+")
            f.write("MODE " + modality + " SEQUENCE " + name +  " QP " +  transl[quality_level] + " BITS " +  str(bpp) + " YPSNR " +  str(metrics["psnr"]) +  " YMSSIM " +  str(metrics["ms-ssim"]) + "\n")
            
        f.close()  
        """
        
    #f=open("/scratch/inference/results/kodak/bjonte/devil2022/FINEGRAINED_0707_kodak_TOTAL.txt" , "a+")
    #f.write("MODE prop" +" SEQUENCE 0 " +  " QP " +  transl_sos[quality_level] + " BITS " +  str(bpps.avg) + " YPSNR " +  str(psnr.avg) +  " YMSSIM " +  str(ms_ssim.avg) + "\n")
    #f.close()      
    print("fine inddferenza",psnr.avg, ms_ssim.avg, bpps.avg)
    return psnr.avg, ms_ssim.avg, bpps.avg


def bpp_calculation(out_net, out_enc):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]

        bpp_1 = (len(out_enc[0]) * 8.0 ) / num_pixels
        #print("la lunghezza è: ",len(out_enc[1]))
        bpp_2 =  sum( (len(out_enc[1][i]) * 8.0 ) / num_pixels for i in range(len(out_enc[1])))
        return bpp_1 + bpp_2, bpp_1, bpp_2



def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

@torch.no_grad()
def eval_models(res, dataloader, device, entropy_estimation, desired_quality = [1,2,3,4,5,6]):
  
    metrics = {}
    models_name = list(res.keys())
    for i, name in enumerate(models_name): #name = q1-bmshj2018-base/fact
        #print("----")
        print("name eval: ",name)
        qual = int(name.split("/")[-1][1])
        model = res[name]["model"]

        sos = True if "base" not in name else False


        if qual in desired_quality: # (1,2,3,5,7,8):#in (2,5,6,8,7,1,3,4,9):
            print("entro nell'inferenza")
            psnr, mssim, bpp =  inference(model,dataloader,device, sos, name, entropy_estimation= entropy_estimation)
            print("qual ",qual,"psnr ",psnr," ",bpp)
            metrics[name] = {"bpp": bpp,
                            "mssim": mssim,
                            "psnr": psnr
                                } 

    return metrics   

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


def load_only_baselines( device, desired_base_quality =  [1,2,3,4,5,6]):
    res = {}

    base_path = "/scratch/pretrained_models/zou22"
    base_models = os.listdir(base_path)
    for bm in base_models:
            
        qual = int(bm.split("-")[0][1:])
        if qual in desired_base_quality:

            pt = os.path.join("/scratch/pretrained_models/zou2022",bm)
            state_dict = load_pretrained(torch.load(pt, map_location=device))
            model = from_state_dict(aux_net_models["cnn"], state_dict) #.eval()
            model.update()
            model.to(device) 

        nome_completo = "q" + str(qual) + "-zou22-base"
        res[nome_completo] = { "model": model}

    return res

def extract_specific_model_performance(metrics, name):

    nms = list(metrics.keys())




    psnr = []
    mssim = []
    bpp = []
    for names in nms:
        if name in names:
            psnr.append(metrics[names]["psnr"])
            mssim.append(metrics[names]["mssim"])
            bpp.append(metrics[names]["bpp"])
    
    return sorted(psnr), sorted(mssim), sorted(bpp)






def export_and_save_results(metrics,save_txt_path):
    pass
     


def main(argv):
    set_seed()
    args = parse_args(argv)
    model_name = args.model  # nome del modello che voglio studiare (ad esempio cheng2020)
    models_path = join(args.model_path,model_name) # percorso completo per arrivare ai modelli salvati (/scratch/inference/pretrained_models/chegn2020) qua ho salvato i modelli 
 

    models_checkpoint =[models_path + "/q5-zou22.pth.tar",models_path + "/q6-zou22.pth.tar",models_path + "/q4-zou22.pth.tar"]# listdir(models_path) # checkpoints dei modelli  q1-bmshj2018-sos.pth.tar, q2-....
    print(models_checkpoint)
    device = "cuda"
    entropy_estimation = args.entropy_estimation
    
    images_path = args.image_path # path del test set 
    #savepath = args.result_path # path dove salvare i risultati 

    image_list = [os.path.join(images_path,f) for f in listdir(images_path)]
    

    dict_model_list =  {} #  inizializzo i modelli 



    for i, check in enumerate(models_checkpoint):  # per ogni cjeckpoint, salvo il modello nostro con la chiave q1-bmshj2018-sos ed il modello base con la chiave q1-bmshj2018-base (il modello base non ha checkpoint perchè lo prendo online)
        if True: #"q1" in check:
            name = check.split("-")[0] + "-" + check.split("-")[1]  # q1-zou22 
            print("name_sos è il seguente: ",name)
            dict_model_list[name + "-sos"] = check
            #dict_model_list[name + "-base"] = name + "-base"
            




    res = load_models(dict_model_list,  models_path, device, image_models) # carico i modelli res è un dict che ha questa struttura res[q1-bmshj2018-sos] = {"model": model}
    #res = load_only_baselines(  "zou22-base", device)

    # cambiato con test_dataloader
    metrics = eval_models(res,image_list , device,entropy_estimation) #faccio valutazione dei modelli 



    # now for every model I have the results, I have only to extract them and write to a file .txt

    #save_txt_path = os.path.join("/scratch/inference/results/RD_curve_files/zou22/",args.dat)
    #export_and_save_results(metrics,save_txt_path, dataset = args.dat)    




    print("ALL DONE!!!!!")




    
if __name__ == "__main__":

    wandb.init(project="prova", entity="albipresta")   
    main(sys.argv[1:])
