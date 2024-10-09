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





#####################################################################################################################################
############################################    CAR #################################################################################
#####################################################################################################################################

x_car_com_only = [2.2271662763466042, 2.901639344262295, 3.899297423887588, 5.6276346604215455, 5.6276346604215455, 8.058548009367682, 11.234192037470725]
y_car_com_only = [78.68537074148297, 79.56713426853707, 79.58717434869739, 79.60721442885772, 79.60721442885772, 86.90180360721443, 79.84769539078157]


x_car_pgr = [1.9837067209775967, 2.5539714867617107, 3.3849287169042768, 4.818737270875763, 6.757637474541751, 9.25050916496945]
y_car_pgr = [78.17919075144509, 78.8728323699422, 78.8728323699422, 78.9306358381503, 84.6242774566474, 79.33526011560693]


x_car_pgr_wo_rest = [1.5386416861826697, 2.0163934426229506, 2.6627634660421546, 3.7447306791569086, 5.177985948477752, 6.878220140515222]
y_car_pgr_wo_rest = [75.71943887775551, 77.20240480961924, 77.72344689378758, 77.66332665330661, 77.92384769539078, 77.66332665330661]



x_car_gndnet = [1.2505091649694502, 1.6415478615071284, 2.130346232179226, 3.0753564154786153, 4.232179226069246, 5.617107942973523]
y_car_gndnet = [67.80346820809248, 74.45086705202313, 74.45086705202313, 74.91329479768785, 74.91329479768785, 75.14450867052022]

x_car_polar = [1.6935483870967742, 2.1451612903225805, 2.838709677419355, 3.9838709677419355, 5.435483870967742, 7.193548387096774]
y_car_polar = [76.6412213740458, 77.84351145038168, 77.78625954198473, 78.07251908396947, 78.15839694656489, 78.2442748091603]


x_car_patch = [1.4516129032258065, 1.8548387096774195, 2.4193548387096775, 3.4193548387096775, 4.64516129032258, 6.145161290322581]
y_car_patch = [76.06870229007633, 77.6145038167939, 77.6145038167939, 77.9293893129771, 77.98664122137404, 78.21564885496183]


#####################################################################################################################################
############################################    PEDESTRIAN #################################################################################
#####################################################################################################################################


x_ped_co = [2.294930875576037, 2.294930875576037, 2.9585253456221197, 3.9400921658986174, 5.654377880184332, 8.059907834101383, 11.184331797235023]
y_ped_co = [38.79245283018868, 38.79245283018868, 49.924528301886795, 52.716981132075475, 58.90566037735849, 59.54716981132076, 58.52830188679245]



x_ped_pgr = [2.032258064516129, 2.6129032258064515, 3.442396313364055, 4.8663594470046085, 6.774193548387097, 9.207373271889402]
y_ped_pgr = [38.490566037735846, 50, 53.35849056603774, 59.056603773584904, 59.20754716981132, 58.52830188679245]


x_ped_pgr_wo = [1.6175115207373272, 2.073732718894009, 2.7096774193548385, 3.7880184331797233, 3.7880184331797233, 5.1981566820276495, 6.8986175115207375]
y_ped_pgr_wo = [38.41509433962264, 49.924528301886795, 53.132075471698116, 58.56603773584905, 58.56603773584905, 58.301886792452834, 58.867924528301884]

x_ped_gndnet = [1.274161735700197, 1.6528599605522682, 2.157790927021696, 3.0887573964497044, 4.240631163708087, 5.597633136094675]
y_ped_gndnet = [20.390243902439025, 29.300813008130085, 34.764227642276424, 37.951219512195124, 38.27642276422765, 37.170731707317074]


x_ped_polar = [1.7159763313609466, 2.1893491124260356, 2.8520710059171597, 3.988165680473373, 5.439842209072978, 7.175542406311638]
y_ped_polar = [36.32520325203252, 46.926829268292686, 50.829268292682926, 54.53658536585366, 52.97560975609756, 52.91056910569106]



x_ped_patch = [1.4635108481262327, 1.873767258382643, 2.4418145956607495, 3.4201183431952664, 4.650887573964497, 6.134122287968442]
y_ped_patch = [31.056910569105693, 41.85365853658537, 48.22764227642277, 51.47967479674797, 52.19512195121951, 51.9349593495935]

#####################################################################################################################################
############################################    CYCLIST #################################################################################
#####################################################################################################################################


x_cy_co = [2.2607449856733526, 2.9312320916905446, 3.8939828080229226, 5.64756446991404, 8.07163323782235, 11.20057306590258]
y_cy_co = [64.56521739130434, 65.32608695652173, 66.34782608695652, 66.83695652173913, 66.68478260869566, 67.36956521739131]


x_cy_pgr = [2.002865329512894, 2.5873925501432664, 3.4126074498567336, 4.839541547277937, 6.765042979942693, 9.206303724928368]
y_cy_pgr = [64.30434782608695, 65.29347826086956, 66.31521739130434, 66.6195652173913, 66.57608695652173, 66.8695652173913]


x_cy_pgr_wo = [1.5902578796561604, 2.037249283667622, 2.69054441260745, 3.773638968481375, 5.1833810888252145, 6.885386819484241]
y_cy_pgr_wo = [62.76086956521739, 65.1413043478261, 65.48913043478261, 65.3695652173913, 65.47826086956522, 65.58695652173913]



x_cy_gnd = [1.3043478260869565, 1.6521739130434783, 2.205533596837945, 3.1067193675889326, 4.260869565217391, 5.66798418972332]
y_cy_gnd = [53.06772908366534, 55.88844621513944, 56.366533864541836, 61.984063745019924, 57.298804780876495, 63.39442231075697]


x_cy_polar = [1.715415019762846, 2.205533596837945, 2.8853754940711465, 3.992094861660079, 5.446640316205533, 7.185770750988142]
y_cy_polar = [61.745019920318725, 65.40239043824701, 64.47011952191235, 64.87649402390439, 65.59362549800797, 65.45019920318725]

x_cy_patch = [1.4782608695652173, 1.8893280632411067, 2.474308300395257, 3.454545454545454, 4.6561264822134385, 6.173913043478261]
y_cy_patch = [61.79282868525897, 65.21115537848605, 64.99601593625498, 64.42231075697211, 65.16334661354581, 66.14342629482071]

print("*************************************************** PGR ********************************************")
print("car")
print('BD-PSNR: ', BD_PSNR(x_car_com_only, y_car_com_only, x_car_pgr,y_car_pgr))
print('BD-RATE: ', BD_RATE(x_car_com_only, y_car_com_only, x_car_pgr, y_car_pgr))


print("Pedestrian")
print('BD-PSNR: ', BD_PSNR(x_ped_co, y_ped_co, x_ped_pgr,y_ped_pgr))
print('BD-RATE: ', BD_RATE(x_ped_co, y_ped_co, x_ped_pgr, y_ped_pgr))


print("Cyclist")
print('BD-PSNR: ', BD_PSNR(x_cy_co, y_cy_co, x_cy_pgr,y_cy_pgr))
print('BD-RATE: ', BD_RATE(x_cy_co, y_cy_co, x_car_pgr, y_cy_pgr))

print("*************************************************** PGR WO REST ********************************************")
print("car")
print('BD-PSNR: ', BD_PSNR(x_car_com_only, y_car_com_only, x_car_pgr_wo_rest,y_car_pgr_wo_rest))
print('BD-RATE: ', BD_RATE(x_car_com_only, y_car_com_only, x_car_pgr_wo_rest, y_car_pgr_wo_rest))


print("Pedestrian")
print('BD-PSNR: ', BD_PSNR(x_ped_co, y_ped_co, x_ped_pgr_wo,y_ped_pgr_wo))
print('BD-RATE: ', BD_RATE(x_ped_co, y_ped_co, x_ped_pgr_wo, y_ped_pgr_wo))


print("Cyclist")
print('BD-PSNR: ', BD_PSNR(x_cy_co, y_cy_co, x_cy_pgr_wo,y_cy_pgr_wo))
print('BD-RATE: ', BD_RATE(x_cy_co, y_cy_co, x_cy_pgr_wo, y_cy_pgr_wo))

print("*************************************************** PATCHWORK ********************************************")
print("car")
print('BD-PSNR: ', BD_PSNR(x_car_com_only, y_car_com_only, x_car_patch,y_car_patch))
print('BD-RATE: ', BD_RATE(x_car_com_only, y_car_com_only, x_car_patch, y_car_patch))


print("Pedestrian")
print('BD-PSNR: ', BD_PSNR(x_ped_co, y_ped_co, x_ped_patch,y_ped_patch))
print('BD-RATE: ', BD_RATE(x_ped_co, y_ped_co, x_ped_patch, y_ped_patch))


print("Cyclist")
print('BD-PSNR: ', BD_PSNR(x_cy_co, y_cy_co, x_cy_patch,y_cy_patch))
print('BD-RATE: ', BD_RATE(x_cy_co, y_cy_co, x_cy_patch, y_cy_patch))

print("*************************************************** GNDNET ********************************************")

print('BD-PSNR: ', BD_PSNR(x_car_com_only[:-2], y_car_com_only[:-2], x_car_gndnet,y_car_gndnet))
print('BD-RATE: ', BD_RATE(x_car_com_only[:-2], y_car_com_only[:-2], x_car_gndnet, y_car_gndnet))


print("Pedestrian")
print('BD-PSNR: ', BD_PSNR(x_ped_co, y_ped_co, x_ped_gndnet,y_ped_gndnet))
print('BD-RATE: ', BD_RATE(x_ped_co, y_ped_co, x_ped_gndnet, y_ped_gndnet))


print("Cyclist")
print('BD-PSNR: ', BD_PSNR(x_cy_co, y_cy_co, x_cy_gnd,y_cy_gnd))
print('BD-RATE: ', BD_RATE(x_cy_co, y_cy_co, x_cy_gnd, y_cy_gnd))



print("*************************************************** POLAR ********************************************")
print("car")
print('BD-PSNR: ', BD_PSNR(x_car_com_only, y_car_com_only, x_car_polar,y_car_polar))
print('BD-RATE: ', BD_RATE(x_car_com_only, y_car_com_only, x_car_polar, y_car_polar))


print("Pedestrian")
print('BD-PSNR: ', BD_PSNR(x_ped_co, y_ped_co, x_ped_polar,y_ped_polar))
print('BD-RATE: ', BD_RATE(x_ped_co, y_ped_co, x_ped_polar, y_ped_polar))


print("Cyclist")
print('BD-PSNR: ', BD_PSNR(x_cy_co, y_cy_co, x_cy_polar,y_cy_polar))
print('BD-RATE: ', BD_RATE(x_cy_co, y_cy_co, x_cy_polar, y_cy_polar))




print("car our vs polar")
print('BD-PSNR: ', BD_PSNR(x_car_polar, y_car_polar, x_car_pgr,y_car_pgr))
print('BD-RATE: ', BD_RATE(x_car_polar, y_car_polar, x_car_pgr, y_car_pgr))

print("car our vs gndnet")
print('BD-PSNR: ', BD_PSNR(x_car_gndnet, y_car_gndnet, x_car_pgr,y_car_pgr))
print('BD-RATE: ', BD_RATE(x_car_gndnet, y_car_gndnet, x_car_pgr, y_car_pgr))
