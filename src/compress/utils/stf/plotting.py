
import torch 
import wandb
import seaborn as sns
palette = sns.color_palette("tab10")
import matplotlib.pyplot as plt


def plot_sos(model, device,n = 1000, dim = 0,sl = 1 ):
    
    x_min = float((min(model.gaussian_conditional[sl].sos.b) + min(model.gaussian_conditional[sl].sos.b)*0.5).detach().cpu().numpy())
    x_max = float((max(model.gaussian_conditional[sl].sos.b)+ max(model.gaussian_conditional[sl].sos.b)*0.5).detach().cpu().numpy())
    step = (x_max-x_min)/n
    x_values = torch.arange(x_min, x_max, step)
    x_values = x_values.repeat(model.gaussian_conditional[sl].M,1,1)
            
    y_values=model.gaussian_conditional[sl].sos(x_values.to(device))[0,0,:]
    data = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
    table = wandb.Table(data=data, columns = ["x", "sos"])
    wandb.log({"GaussianSoS/Gaussian SoS at dimension " + str(dim): wandb.plot.line(table, "x", "sos", title='GaussianSoS/Gaussian SoS  with beta = {}'.format(model.gaussian_conditional.sos.beta))})
    y_values= model.gaussian_conditional[sl].sos(x_values.to(device), -1)[0,0,:]
    data_inf = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
    table_inf = wandb.Table(data=data_inf, columns = ["x", "sos"])
    wandb.log({"GaussianSoS/Gaussian SoS  inf at dimension " + str(dim): wandb.plot.line(table_inf, "x", "sos", title='GaussianSoS/Gaussian SoS  with beta = {}'.format(-1))})  



def plot_rate_distorsion(bpp_res, psnr_res,epoch, eest = "compression", index_list = []):

    chiavi_da_mettere = list(psnr_res.keys())
    legenda = {}
    for i,c in enumerate(chiavi_da_mettere):
        legenda[c] = {}
        legenda[c]["colore"] = [palette[i],'-']
        legenda[c]["legends"] = c
        legenda[c]["symbols"] = ["*"]*300
        legenda[c]["markersize"] = [5]*300    

    plt.figure(figsize=(12,8)) # fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    list_names = list(psnr_res.keys())

    minimo_bpp, minimo_psnr = 10000,1000
    massimo_bpp, massimo_psnr = 0,0

    for _,type_name in enumerate(list_names): 

        bpp = bpp_res[type_name]
        psnr = psnr_res[type_name]
        colore = legenda[type_name]["colore"][0]
        #symbols = legenda[type_name]["symbols"]
        #markersize = legenda[type_name]["markersize"]
        leg = legenda[type_name]["legends"]

        bpp = torch.tensor(bpp).cpu()
        psnr = torch.tensor(psnr).cpu()    
        plt.plot(bpp,psnr,"-" ,color = colore, label =  leg ,markersize=8)       
        plt.plot(bpp, psnr, marker="o", markersize=4, color =  colore)


        for j in range(len(bpp)):
            if bpp[j] < minimo_bpp:
                minimo_bpp = bpp[j]
            if bpp[j] > massimo_bpp:
                massimo_bpp = bpp[j]
            
            if psnr[j] < minimo_psnr:
                minimo_psnr = psnr[j]
            if psnr[j] > massimo_psnr:
                massimo_psnr = psnr[j]

    minimo_psnr = int(minimo_psnr)
    massimo_psnr = int(massimo_psnr)
    psnr_tick =  [round(x) for x in range(minimo_psnr, massimo_psnr + 2)]
    plt.ylabel('PSNR', fontsize = 30)
    plt.yticks(psnr_tick)

    #print(minimo_bpp,"  ",massimo_bpp)

    bpp_tick =   [round(x)/10 for x in range(int(minimo_bpp*10), int(massimo_bpp*10 + 2))]
    plt.xticks(bpp_tick)
    plt.xlabel('Bit-rate [bpp]', fontsize = 30)
    plt.yticks(fontsize=27)
    plt.xticks(fontsize=27)
    plt.grid()

    plt.legend(loc='lower right', fontsize = 25)



    plt.grid(True)
    if eest == "model":
        wandb.log({"model":epoch,
              "model/rate distorsion trade-off": wandb.Image(plt)})
    else:  
        wandb.log({"compression":epoch,
              "compression/rate distorsion trade-off": wandb.Image(plt)})       
    plt.close()  
    print("FINITO")