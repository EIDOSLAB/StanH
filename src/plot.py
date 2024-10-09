

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

def plot_rate_distorsion(bpp_res, psnr_res,nome_file,color, index_list = None):

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

        
        if "VTM" not in type_name:
            plt.plot(bpp,psnr,"-" ,color = colore, label =  leg ,markersize=10)       
            plt.plot(bpp, psnr, marker="o", markersize=10, color =  colore)
        else:
            plt.plot(bpp,psnr,"-." ,color = colore, label =  leg ,markersize=11)  




        if "Gain" not in type_name and "DCVC" not in type_name and "EVC" not in type_name and "VTM" not in type_name and "Reference" not in type_name and "SCR" not in type_name and "Manual" not in type_name:
           
            index_list = [] if index_list is None else index_list
            for jjj in index_list:
                plt.plot(bpp[jjj], psnr[jjj], marker="X", markersize=17, color =  colore)
                #plt.plot(bpp[jjj], psnr[jjj], marker="*", markersize=8, color =  colore)
               # plt.plot(bpp[jjj], psnr[jjj], marker="*", markersize=8, color =  colore) #fff


        if "Manual"  in type_name:
           
            index_list = [2] 
            for jjj in index_list:
                plt.plot(bpp[jjj], psnr[jjj], marker="X", markersize=17, color =  colore)
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

    minimo_psnr = 29 #int(minimo_psnr)
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




    bpp_res["Zou22 + GainUnits"] = [0.17, 0.23839285714285716, 0.3410714285714286, 0.47410714285714284, 0.6321428571428571, 0.8133928571428571] #0.17, 0.23839285714285716,
    psnr_res["Zou22 + GainUnits"] = [29.372,30.805755395683455, 32.34532374100719, 33.94244604316547, 35.29496402877698, 36.460431654676256] #29.372,30.805755395683455
    color["Zou22 + GainUnits"] = 8



    
    bpp_res["EVC"] = [     0.3311998920792,
        0.509808708,
        0.66811443933844567,
        0.758309958,]
    psnr_res["EVC"] =      [          32.38911329,
        34.39935488,
        35.64286368370056,
        36.30560738,
       ]
    
    color["EVC"] = 0
    

    
    #ANCORA A 0.400
    bpp_res["Zou22 + StanH (our)"] = [ 0.18,0.2288, 0.26, 0.2912, 0.3328, 0.40, 0.4714, 0.5809, 0.6509, 0.778] #0.198 
    psnr_res["Zou22 + StanH (our)"] = [28.97565,30.72, 31.34, 31.904, 32.630, 33.54, 34.214, 34.901, 35.17, 35.4565] #29.735
    index_list  = [5]
    color["Zou22 + StanH (our)"] = 3
    



    

    #ANCORA A 0.600
    #bpp_res["Zou22 + StanH (our)"] = [0.3338983050847458, 0.3990466101694915, 0.4281779661016949, 0.4588,0.5144, 0.555,0.59480, 0.67433, 0.7268292682926829, 0.7878048780487805]  
    #psnr_res["Zou22 + StanH (our)"] = [31.641221374045802, 32.83206106870229, 33.22595419847328, 33.638167938931296, 34.34,34.72824427480916,35.17629179331307, 35.74164133738602, 36.1063829787234, 36.41267875]
    #index_list  = [4]


    #ANCORE A 0.4-0.84 
    #bpp_res["Zou22 + StanH (our)"] = [0.18,0.2288, 0.26, 0.2912, 0.3328, 0.40, 0.4714, 0.5909, 0.6149,0.7324715615305067, 0.7841778697001034,0.8403369672943508, 0.9144777662874871]  
    #psnr_res["Zou22 + StanH (our)"] = [29.03,30.72, 31.34, 31.904, 32.630, 33.54, 34.214, 34.901, 35.185,36.345238095238095, 36.773809523809526, 37.23520923520923, 37.68452380952381]
    #index_list  = [5,11]
    #color["Zou22 + StanH (our)"] = 3

    #ANCORE A 0.4-0.6 
    #bpp_res["Zou22 + StanH (our)"] = [0.2388, 0.26, 0.2912, 0.3328, 0.40, 0.4714, 0.5948038176033934, 0.6743372216330858, 0.7268292682926829, 0.7878048780487805]  
    #psnr_res["Zou22 + StanH (our)"] = [30.72, 31.34, 31.904, 32.630, 33.54, 34.214, 35.17629179331307, 35.74164133738602, 36.1063829787234, 36.41267875]
    #index_list  = [4,6]






    #ANCORE A 0.14-0.4-0.6 
    #bpp_res["Zou22 + StanH (our)"] = [0.10743801652892562, 0.12396694214876033, 0.14132231404958678, 0.15123966942148762, 0.1706611570247934,0.20,0.2413,0.26, 0.2912, 0.3328, 0.40, 0.4714, 0.5948038176033934, 0.6743372216330858, 0.7268292682926829, 0.7878048780487805]  
    #psnr_res["Zou22 + StanH (our)"] = [27.762295081967213, 28.524590163934427, 29.065573770491802, 29.348360655737704, 29.729508196721312, 30.13,30.72, 31.34, 31.904, 32.630, 33.54, 34.214, 35.17629179331307, 35.74164133738602, 36.1063829787234, 36.41267875]
    #index_list  = [1,10,12]
    #color["Zou22 + StanH (our)"] = 3


    #3ANCORE A 0.014-0.4-0.84 
    #bpp_res["Zou22 + StanH (our)"] = [0.1074, 0.1239, 0.141, 0.151, 0.1706,0.20, 0.22, 0.26, 0.2912, 0.3328, 0.40, 0.4714, 0.5909, 0.6149,   0.7324715615305067, 0.7841778697001034,0.8403369672943508, 0.9144777662874871]  
    #psnr_res["Zou22 + StanH (our)"] = [27.7622, 28.524, 29.065, 29.3483, 29.729, 30.13,  30.28, 31.34, 31.904, 32.630, 33.54, 34.214, 34.901, 35.185,   36.345238095238095, 36.773809523809526, 37.23520923520923, 37.68452380952381]
    #index_list  = [2,10,16]
    #color["Zou22 + StanH (our)"] = 3



    #3ANCORE A 0.4-0.6-0.84 
    #bpp_res["Zou22 + StanH (our)"] = [0.18,0.2288, 0.26, 0.2912, 0.3328, 0.40, 0.4714,0.5144, 0.555,0.59480, 0.67433,  0.7324715615305067, 0.7841778697001034,0.8403369672943508, 0.9144777662874871]  
    #psnr_res["Zou22 + StanH (our)"] = [29.03,30.72, 31.34, 31.904, 32.630, 33.54, 34.214,34.34,34.7282,35.176, 35.741,36.345238095238095, 36.773809523809526, 37.23520923520923, 37.68452380952381]
    #index_list  = [5,9,13]
    color["Zou22 + StanH (our)"] = 3







    #bpp_res["Zou22 + Manual"] = [0.2356513222331048, 0.31860920666013715,  0.40, 0.4852105778648384, 0.575024485798237, 0.6888344760039178, 0.8389813907933399]
    #psnr_res["Zou22 + Manual"] = [30.37278106508876, 32.36094674556213, 33.54,34.25147928994083, 34.71301775147929, 35.023668639053255, 35.17455621301775]
    #color["Zou22 + Manual"] = 2



    # ANCORA A 0.6 BPP
    #bpp_res["Zou22-1A-StanH (our)"] = [0.3339342523860021, 0.4012725344644751, 0.45959703075291625, 0.5948038176033934, 0.6743372216330858, 0.7268292682926829, 0.7878048780487805]
    #psnr_res["Zou22-1A-StanH (our)"] = [31.6401, 32.851063829787236, 33.62613981762918, 35.17629179331307, 35.74164133738602, 36.1063829787234, 36.41267875]

    #bpp_res["DCVC"] =  [0.3288895665771432, 0.40111285613642794, 0.48183066563473814, 0.5708754377232658, 0.6661443933844567, 0.7668840206331677]#, 0.871793516808086, 0.9805055922932095]
    #psnr_res["DCVC"] = [32.3681632677714, 33.264776865641274, 34.1336555480957, 34.94919363657633, 35.71286368370056, 36.40998935699463]#, 37.017213662465416, 37.54074923197428]
    #color["DCVC"] = 7
        


    bpp_res["SCR"] = [ 0.18795620437956204, 0.21350364963503649, 0.2572992700729927, 0.31934306569343063, 0.3886861313868613, 0.46715328467153283, 0.5218978102189781, 0.6240875912408759, 0.6715328467153284, 0.7481751824817519, 0.8211678832116788 ]
    psnr_res["SCR"] = [ 29.334016393442624, 29.764344262295083, 30.563524590163937, 31.454918032786885, 32.3155737704918, 33.20696721311475, 33.760245901639344, 34.743852459016395, 35.08196721311475, 35.75819672131148, 36.28073770491803]
    color["SCR"] = 6
    #name_file = "prova.png"




    #0.1386804657179819, 0.15109961190168175, 0.17283311772315654, 0.19560155239327298,0.192496765847348, 0.20802069857697283, 0.24113842173350583, 0.27632600258732215
    #29.046676096181045, 29.414427157001413, 29.90947666195191, 30.291371994342292,30.22065063649222, 30.531824611032533, 31.055162659123056, 31.64922206506365

    bpp_res["VTM"]  = [0.175,0.2486786, 0.32496765847347997, 0.37050452781371285, 0.415006468305304, 0.4553686934023286, 0.48952134540750325, 0.5319534282018111, 0.583699870633894, 0.6292367399741268, 0.6675291073738681, 0.7058214747736093, 0.7523932729624838, 0.7865459249676585]
    psnr_res["VTM"] = [29.75,31.1845646464, 32.2998585572843, 32.90806223479491, 33.417256011315416, 33.88401697312588, 34.22347949080623, 34.633663366336634, 35.11456859971712, 35.510608203677506, 35.807637906647805, 36.13295615275813, 36.5007072135785, 36.75530410183875]

    #bpp_res["VTM"] = [0.2, 0.24,0.4915094339622642, 0.630188679245283,0.87]
    #psnr_res["VTM"] = [30.40,  31.198, 34.2600422832981, 35.42706131078224,37.41] 
    color["VTM"] = 7
    



    name_file = "1A_VTM.png"
    plot_rate_distorsion(bpp_res,psnr_res, name_file,color,index_list=index_list)









    
    print("Our-bd*")
    print('BD-PSNR: ', BD_PSNR(bpp_res["VTM"], psnr_res["VTM"], bpp_res["Zou22 + StanH (our)"],psnr_res["Zou22 + StanH (our)"]))
    print('BD-RATE: ', BD_RATE(bpp_res["VTM"], psnr_res["VTM"], bpp_res["Zou22 + StanH (our)"], psnr_res["Zou22 + StanH (our)"]))

    print("EVC")
    print('BD-PSNR: ', BD_PSNR(bpp_res["VTM"], psnr_res["VTM"], bpp_res["EVC"],psnr_res["EVC"]))
    print('BD-RATE: ', BD_RATE(bpp_res["VTM"], psnr_res["VTM"], bpp_res["EVC"], psnr_res["EVC"]))


    print("gaIN")
    print('BD-PSNR: ', BD_PSNR(bpp_res["VTM"], psnr_res["VTM"], bpp_res["Zou22 + GainUnits"],psnr_res["Zou22 + GainUnits"]))
    print('BD-RATE: ', BD_RATE(bpp_res["VTM"], psnr_res["VTM"], bpp_res["Zou22 + GainUnits"], psnr_res["Zou22 + GainUnits"]))



    print("Our-bd*-EVC")
    print('BD-PSNR: ', BD_PSNR(bpp_res["EVC"],psnr_res["EVC"], bpp_res["Zou22 + StanH (our)"],psnr_res["Zou22 + StanH (our)"]))
    print('BD-RATE: ', BD_RATE(bpp_res["EVC"],psnr_res["EVC"], bpp_res["Zou22 + StanH (our)"], psnr_res["Zou22 + StanH (our)"]))
    
    print("Our-bd*-GAIN")
    print('BD-PSNR: ', BD_PSNR(bpp_res["Zou22 + GainUnits"],psnr_res["Zou22 + GainUnits"], bpp_res["Zou22 + StanH (our)"],psnr_res["Zou22 + StanH (our)"]))
    print('BD-RATE: ', BD_RATE(bpp_res["Zou22 + GainUnits"],psnr_res["Zou22 + GainUnits"], bpp_res["Zou22 + StanH (our)"], psnr_res["Zou22 + StanH (our)"]))
    
    


if __name__ == "__main__":

     
    main()