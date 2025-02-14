import argparse

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    
    parser.add_argument("-mp","--anchor_path",default="/scratch/inference/new_models/devil2022/",help="Model architecture (default: %(default)s)",)
    parser.add_argument("--device",default="cuda",help="device (cuda or cpu)",)
    parser.add_argument("--wandb_log", action="store_true", help="Use cuda")
    parser.add_argument("--activation",default="nonlinearstanh",type=str,help="factorized_annealing",)
    parser.add_argument("--lmbda", nargs='+', type=float, default =[0.025])
    
    parser.add_argument("-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",)

    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    parser.add_argument("-ni","--num_images",default = 48064, type = int)
    parser.add_argument("-niv","--num_images_val",default = 1024, type = int)
    parser.add_argument("-ex","--extrema",default = 60, type = int)

    parser.add_argument("-sp","--stanh_path",default="/scratch/inference/new_models/devil2022/3_anchors_stanh",help="Model architecture (default: %(default)s)",)#dddd
    
    parser.add_argument("-ip","--image_path",default="/scratch/dataset/kodak",help="Model architecture (default: %(default)s)",)
    parser.add_argument("--entropy_estimation", action="store_true", help="Use cuda")
    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--pretrained_stanh", action="store_true", help="Use cuda")
    parser.add_argument("--only_dist", action="store_true", help="Use cuda")
    parser.add_argument("--unfreeze_fact", action="store_true", help="Use cuda")
    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--filename",default="/data/",type=str,help="factorized_annealing",)
    
    parser.add_argument("-e","--epochs",default=600,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument("--gauss_gp",default=15,type=int,help="gauss_beta",)
    parser.add_argument("--gauss_annealing",default="gap_stoc",type=str,help="factorized_annealing",)
    parser.add_argument("--anchor_path",default="/scratch/pretrained_models/stf/stf_013.pth.tar",type=str,help="factorized_annealing",)
    
    parser.add_argument("--num_stanh", type=int, default=1, help="Batch size (default: %(default)s)")
    parser.add_argument("--training_focus",default="stanh_levels",type=str,help="factorized_annealing",)

    args = parser.parse_args(argv) ###s
    return args