

import seaborn as sns
palette = sns.color_palette("tab10")
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt

palette = sns.color_palette("tab10")
#plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman')

import numpy as np
import scipy.interpolate


def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)

    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), np.sort(PSNR1), samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), np.sort(PSNR2), samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_diff = (int2-int1)/(max_int-min_int)

    return avg_diff


def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), np.sort(lR1), samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), np.sort(lR2), samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2-int1)/(max_int-min_int)
    avg_diff = (np.exp(avg_exp_diff)-1)*100
    return avg_diff

def plot_rate_distorsion(bpp_res, psnr_res,nome_file,color, index_list = None, index_stanh = None):

    chiavi_da_mettere = list(psnr_res.keys())
    legenda = {}
    for i,c in enumerate(chiavi_da_mettere):
        legenda[c] = {}
        legenda[c]["colore"] = [palette[color[c]],'-']
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

        
        if "StanH" not in type_name:
            plt.plot(bpp,psnr,"-." ,color = colore, label =  leg ,markersize=10)      
            for iii in range(len(bpp)): 
                plt.plot(bpp[iii], psnr[iii], marker="X", markersize=18, color =  colore)
        else:
            plt.plot(bpp,psnr,"-" ,color = colore, label =  leg ,markersize=10)       
            plt.plot(bpp, psnr, marker="s", markersize=4, markerfacecolor='none', color =  colore)            




        if "StanH" in type_name:
           
            index_list = [] if index_list is None else index_list
            for jjj in index_list:
                plt.plot(bpp[jjj], psnr[jjj], marker="X", markersize=18, color =  colore)
            for jjj in index_stanh:
                plt.plot(bpp[jjj], psnr[jjj], marker="o", markersize=10, color =  colore)
                #plt.plot(bpp[jjj], psnr[jjj], marker="*", markersize=8, color =  colore)
               # plt.plot(bpp[jjj], psnr[jjj], marker="*", markersize=8, color =  colore) #fff




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
    psnr_tick =  [round(x) for x in range(minimo_psnr, massimo_psnr + 1)]
    plt.ylabel('PSNR', fontsize = 30)
    plt.yticks(psnr_tick)

    #print(minimo_bpp,"  ",massimo_bpp)

    bpp_tick =   [round(x)/10 for x in range(int(minimo_bpp*10), int(massimo_bpp*10 + 1))]
    plt.xticks(bpp_tick)
    plt.xlabel('Bit-rate [bpp]', fontsize = 30)
    plt.yticks(fontsize=27)
    plt.xticks(fontsize=27)
    plt.grid()

    plt.legend(loc='lower right', fontsize = 25)



    plt.grid(True)
    plt.savefig("../rebuttal/" + nome_file)    
    plt.close()  
    print("FINITO")





def main():

    psnr_res = {}
    bpp_res = {}
    color = {}







    x = [0.6133928571428572, 0.6294642857142857, 0.6473214285714286, 0.6642857142857144, 
         0.6776785714285715, 0.6901785714285714, 0.7044642857142858, 0.7160714285714286, 0.7214285714285714, 0.7312500000000001, 0.7455357142857143, 0.7589285714285715, 0.7714285714285715, 0.78125, 0.7964285714285715, 0.8098214285714286, 0.8214285714285715, 0.83125, 0.8375, 0.8508928571428572, 0.8651785714285715, 0.8812500000000001, 0.8946428571428572, 0.911607142857143]
    y = [35.1740614334471, 35.37883959044369, 35.58361774744027, 35.737201365187715, 
         35.89078498293515, 35.99317406143345, 36.14675767918089, 36.249146757679185, 36.31740614334471, 36.351535836177476, 36.47098976109215, 36.62457337883959, 36.7098976109215, 36.77815699658703, 36.91467576791809, 37.01706484641638, 37.11945392491468, 37.20477815699659, 37.255972696245735, 37.32423208191126, 37.44368600682594, 37.51194539249147, 37.61433447098976, 37.68259385665529]






    bpp_res = {}
    psnr_res = {}



    bpp_res["Zou22"] = [0.114489141675284385, 0.18521199586349535, 0.3002068252326784, 0.46153050672182006, 0.644364012409514, 0.9033092037228543]
    psnr_res["Zou22"] = [28.805072463768117, 30.333333333333332, 32.05797101449275, 33.869565217391305, 35.69565217391305, 37.57971014492754]


    bpp_res["Zou22 + StanH (our)"] = [0.10714285714285715, 0.11607142857142858, 0.1205357142857143, 0.12410714285714286, 0.13125,0.141000      ,0.15, 0.160, 0.171, 0.1848, 0.199,
         0.21, 0.23, 0.243, 0.259, 0.270, 0.278, 0.291, 0.301, 0.314, 0.33, 0.324, 0.34,0.35357142857142865, 0.3723214285714286, 0.3883928571428572, 0.392, 0.405,
         0.4080, 0.4214, 0.4339, 0.446425, 0.45447, 0.4633, 0.4789787, 0.48750, 0.501785, 0.5169, 0.53125, 0.54642, 0.5607142, 0.571428, 0.5838, 0.5892] + x
    psnr_res["Zou22 + StanH (our)"] = [27.733788395904437, 28.109215017064848, 28.348122866894197, 28.51877133105802, 28.74061433447099,29.065333,29.33, 29.54, 29.73, 29.91, 30.122,
         30.29351, 30.686, 31.010, 31.3344, 31.5563, 31.74402, 31.9146, 32.17064, 32.358, 32.665, 32.511, 32.80207,32.93856655290102, 33.22866894197952, 33.43344709897611, 33.48, 33.54, 
         33.65529010238908, 33.774744027303754, 33.894197952218434, 34.04778156996587, 34.081911, 34.133105, 34.2013651, 34.33437, 34.42320828, 34.52559624576, 34.57679180887372, 
         34.67918088737201, 34.764505393, 34.81569075, 34.88386, 34.901084984] + y



    print("----------> psnr ",np.array(psnr_res["Zou22 + StanH (our)"]).mean())
    print("----------> avg ",np.array(bpp_res["Zou22 + StanH (our)"]).mean())

    print(len(psnr_res["Zou22 + StanH (our)"]))

    bpp_res["VTM"] = [0.11, 0.24,0.4915094339622642, 0.630188679245283,0.87]
    psnr_res["VTM"] = [28.49,  31.198, 34.2600422832981, 35.42706131078224,37.41] 
    color["VTM"] = 9

    color["Zou22 + StanH (our)"] = 3
    color["Zou22"] = 0



    print("StanH")
    print('BD-PSNR: ', BD_PSNR(bpp_res["VTM"], psnr_res["VTM"], bpp_res["Zou22 + StanH (our)"],psnr_res["Zou22 + StanH (our)"]))
    print('BD-RATE: ', BD_RATE(bpp_res["VTM"], psnr_res["VTM"], bpp_res["Zou22 + StanH (our)"], psnr_res["Zou22 + StanH (our)"]))

    print("Zou22")
    print('BD-PSNR: ', BD_PSNR(bpp_res["VTM"], psnr_res["VTM"], bpp_res["Zou22"],psnr_res["Zou22"]))
    print('BD-RATE: ', BD_RATE(bpp_res["VTM"], psnr_res["VTM"], bpp_res["Zou22"], psnr_res["Zou22"]))

    print("Zou22")
    print('BD-PSNR: ', BD_PSNR(bpp_res["Zou22"], psnr_res["Zou22"], bpp_res["Zou22 + StanH (our)"],psnr_res["Zou22 + StanH (our)"]))
    print('BD-RATE: ', BD_RATE(bpp_res["Zou22"], psnr_res["Zou22"], bpp_res["Zou22 + StanH (our)"], psnr_res["Zou22 + StanH (our)"]))



    index_list_stanh = [0,2,5,10,11,13,16,20,34,43,44,53,57,67]
    index_list = [5,27,62]


    name_file = "grained.pdf"
    #plot_rate_distorsion(bpp_res,psnr_res, name_file,color,index_list=index_list, index_stanh = index_list_stanh)





if __name__ == "__main__":

     
    main()