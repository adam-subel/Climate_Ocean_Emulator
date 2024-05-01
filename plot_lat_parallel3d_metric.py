import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

import cartopy.crs as ccrs
import cartopy as cart
import cmocean
import torch
import torch.nn as nn
import torch.utils.data as data
import torch_geometric
from torch.nn import Sequential as Seq, Linear, ReLU
from Networks import *
from Data_Functions import *
from matplotlib.animation import FuncAnimation
from Utils import *
from Subgrid_Funcs import *
import torch.distributed as dist

from Parallel import *
from Eval_Funcs import *
from torch.utils.data import Dataset, DataLoader
import os 
import sys
import random

import numpy.fft as fft
import sys
import copy_things as copy
import scipy as sp

import matplotlib.ticker as ticker

os.environ['MKL_THREADING_LAYER'] = 'GNU'

exp_num_in = "3"
exp_num_extra = "12"
exp_num_out = "2"


mse = torch.nn.MSELoss()
args = {}

region = "Gulf_Stream_Ext"  
depth_list = ["0","99","253"]
# 99 253 525
network = "U_net"

interval = 1

N_samples = 1000
N_val = 10
N_test = 400

factor = 10

hist = 0

lag = 1

lam = 100

steps = 4
Nb = 4

rand_seed = 1





if len(sys.argv) > 4:
    n_cond = int((len(sys.argv)-4)/2)

str_video = ""

try:
    for i in range(n_cond):
        if type(globals()[sys.argv[int(4 + i*2)]]) == str:
            temp = str(sys.argv[int(5 + i*2)])
            exec(sys.argv[int(4 + i*2)] +"= temp" )
            if sys.argv[int(4 + i*2)] == "network":
                continue            
            str_video += "_" + sys.argv[int(4 + i*2)] + "_" + sys.argv[int(5 + i*2)]
        elif type(globals()[sys.argv[int(4 + i*2)]]) == int:
            exec(sys.argv[int(4 + i*2)] +"=" + "int(" + sys.argv[int(5 + i*2)] +")" )
            str_video += "_" + sys.argv[int(4 + i*2)] + "_" + sys.argv[int(5 + i*2)]
    print(str_video)
except:
    print("no cond")

str_video += "no_KE"    
    
torch.manual_seed(rand_seed)
random.seed(rand_seed)
np.random.seed(rand_seed)

    
args["Nb"] = Nb    
args["region"] = region
args["network"] = network
args["interval"] = interval
args["N_samples"] = N_samples
args["N_val"] = N_val
args["N_test"] = N_test
args["factor"] = factor
args["hist"] = hist
args["lag"] = lag
args["steps"] = steps
args["str_video"] = str_video

    
if region == "Kuroshio":
    lat = [15,41]
    lon = [-215, -185]
elif region == "Kuroshio_Ext":
    lat = [5,50]
    lon = [-250, -175]      
elif region == "Gulf_Stream":
    lat = [25, 50]
    lon = [-70,-35]
elif region == "Gulf_Stream_Ext":
    lat = [27, 50]
    lon = [-82,-35]       
elif region == "Tropics":
    lat = [-5,25]
    lon = [-95,-65]  
elif region == "Tropics_Ext":
    lat = [-5,25]
    lon = [-115,-45]     
elif region == "South_America":
    lat = [-60, -30]
    lon = [-70,-35] 
elif region == "Africa":
    lat = [-50, -20]
    lon = [5,45] 
elif region == "Quiescent":
    lat = [-42.5, -17.5]
    lon = [-155,-120] 
elif region == "Quiescent_Ext":
    lat = [-55, -10]
    lon = [-170,-110]            
elif region == "Pacific":
    lat = [-35, 35]
    lon = [-230,-80]     
elif region == "Indian":
    lat = [-30, 28]
    lon = [30,79]    
elif region == "Africa_Ext":
    lat = [-55, -15]
    lon = [-5,55]     
    
    
s_train = lag*hist
e_train = s_train + N_samples*interval
e_test = e_train + interval*N_val



device = "cpu"


inpt_dict = {"1":["um","vm"],"2":["um","vm","ur","vr"],"3":["um","vm","Tm"],
            "4":["um","vm","ur","vr","Tm","Tr"],"5":["ur","vr"],"6":["ur","vr","Tr"],
            "7":["Tm"],"8":["Tm","Tr"],"9":["u","v"],"10":["u","v","T"],
            "11":["tau_u","tau_v"],"12":["tau_u","tau_v","t_ref"]} 
extra_dict = {"1":["ur","vr"],"2":["ur","vr","Tm"],
            "3":["Tm"],"4":["ur","vr","Tm","Tr"],"5":[],"6":["um","vm"],
             "7":["um","vm","Tm"], "8": ["um","vm","Tm","Tr"],
              "9":["ur","vr","tau_u","tau_v"],"10":["tau_u","tau_v"],
              "11":["ur","vr","Tr"],"12":["tau_u","tau_v","t_ref"],
             "13":["ur","vr","Tr","tau_u","tau_v","t_ref"]} 
out_dict = {"1":["um","vm"],"2":["um","vm","Tm"],"3":["ur","vr"],
           "4":["ur","vr","Tr"],"5":["u","v"],"6":["u","v","T"]}

'''
grids = xr.open_dataset('/scratch/as15415/Data/CM2x_grids/Grid_cm26_Vertices.nc')

if region == "global_25":
    grids = xr.open_dataset('/scratch/as15415/Data/CM2x_grids/Grid_cm25_Vertices.nc')

elif "global" in region:
    grids = coarse_grid(grids,factor)

else:
    grids = grids.sel({"yu_ocean":slice(lat[0],lat[1]),"xu_ocean":slice(lon[0],lon[1])})


area = torch.from_numpy(grids["area_C"].to_numpy()).to(device=device)
'''
grids = xr.open_dataset('/scratch/as15415/Data/CM2x_grids/Grid_cm25_Vertices.nc')
if "global" in region:
    grids = coarse_grid(grids,factor)

else:
    grids = grids.sel({"yu_ocean":slice(lat[0],lat[1]),"xu_ocean":slice(lon[0],lon[1])})


area = torch.from_numpy(grids["area_C"].to_numpy()).to(device=device)
dx = grids["dxu"].to_numpy()
dy = grids["dyu"].to_numpy()
dy = dy/dx.max()
dx = torch.from_numpy(dx/dx.max()).type(torch.float32)
dy = torch.from_numpy(dy).type(torch.float32)  


inputs_str = inpt_dict[exp_num_in]
extra_in_str = extra_dict[exp_num_extra]
outputs_str = out_dict[exp_num_out]

str_in = "".join([i + "_" for i in inputs_str])
str_ext = "".join([i + "_" for i in extra_in_str])
str_out = "".join([i + "_" for i in outputs_str])

print("inputs: " + str_in)
print("extra inputs: " + str_ext)
print("outputs: " + str_out)

N_atm = len(extra_in_str)
N_var = len(inputs_str)
N_in = (len(inputs_str)*len(depth_list))
N_extra = N_atm + N_in
N_out = int(len(outputs_str)*len(depth_list))

num_in = int((hist+1)*N_in + N_extra)

input_depth = []
output_depth = []

for depth in depth_list:
    for i in range(len(inputs_str)):
        input_depth.append(inputs_str[i]+"_"+depth)
    for i in range(len(outputs_str)):
        output_depth.append(outputs_str[i]+"_"+depth)

norm_vals = get_norms(region,input_depth,extra_in_str,output_depth)

inputs, extra_in, outputs = gen_data_025_lateral_3D(inputs_str,extra_in_str,outputs_str,depth_list,lag,lat,lon,Nb,run_type="")
inputs_2, extra_in_2, outputs_2 = gen_data_025_lateral_3D(inputs_str,extra_in_str,outputs_str,depth_list,lag,lat,lon,Nb,run_type="2x")

wet_list = []
wet_bool_list = []
wet_lap_list = []
wet_lap_bool_list = []
for i in range(len(depth_list)):
    wet = xr.zeros_like(inputs[0][0])
    # inputs[0][0,12,12] = np.nan
    for data in inputs[N_var*i:N_var*(i+1)]:
        wet +=np.isnan(data[0])
    wet_nan = xr.where(wet!=0,np.nan,1).to_numpy()    
    wet = np.isnan(xr.where(wet==0,np.nan,0))
    wet = np.nan_to_num(wet.to_numpy())
    wet = torch.from_numpy(wet).type(torch.float32).to(device="cpu")
    wet_bool = np.array(wet.cpu()).astype(bool)
    wet_bool_list.append(wet_bool)

    wet_lap = compute_laplacian_wet(wet_nan,Nb)
    wet_lap = xr.where(wet_lap==0,1,np.nan)
    wet_lap = np.nan_to_num(wet_lap)
    wet_lap = torch.from_numpy(wet_lap).type(torch.float32).to(device=device)
    wet_list.append(wet)
    wet_lap_list.append(wet_lap)
    wet_lap_bool = np.array(wet_lap).astype(bool)
    wet_lap_bool_list.append(wet_lap_bool)

time_vec = inputs[0].time.data

# 73 is from trial and error
clim = np.zeros((73,*wet.shape,N_out))
for i in range(N_out):
    clim[:,:,:,i] = outputs[i].groupby('time.dayofyear').mean('time').data
  

args["s_train"] = s_train
args["e_train"] = e_train
args["e_test"] = e_test


args["inputs"] = [inputs, inputs_2]
args["extra_in"] = [extra_in, extra_in_2]
args["outputs"] = [outputs, outputs_2]


args["wet"] = wet_list
args["dx"] = dx
args["dy"] = dy
args["wet_lap"] = wet_lap
args["area"] = area
args["N_extra"] = N_extra
args["N_in"] = N_in
args["N_out"] = N_out
args["N_var"] = N_var
args["N_atm"] = N_atm
args["num_in"] = num_in
args["str_in"] = str_in
args["str_ext"] = str_ext 
args["str_out"] = str_out
args["norm_vals"] = norm_vals

args["lateral"] = True
os.environ['MASTER_ADDR'] = 'localhost' 
os.environ['MASTER_PORT'] = str(np.random.randint(1000,1200)) 


data_in_val = gen_data_in(0,e_train,e_test,interval,lag,hist,inputs,extra_in)  
data_out_val = gen_data_out(0,e_train,e_test,lag,interval,outputs)

val_data = data_CNN_Lateral(data_in_val,data_out_val,wet_list,N_atm,Nb,device=device,N_vars=N_var)   

args ["load"] = True

# args["train_data"] = train_data
args["val_data"] = val_data
args["save_model"] = True
args["World_Size"] = int(torch.cuda.device_count())
args["epochs"] = 10
args["batch_size"] = 6
args["lateral"] = True
args["loss_type"] = "KE"
# lam = lam/1000
args["lam"] = 0
# args["step_weights"] = [[1],[.2,.8],[.2,.2,.6],[.1,.1,.2,.8],[.1,.1,.1,.2,.7],[.1,.1,.1,.1,.2,.7],[.1,.1,.1,.1,.1,.2,.7]]
args["step_weights"] = [[1],[1,1],[1,1,1],[1,1,1,1],[1,1,1,1,1],[.1,.1,.1,.1,.2,.7],[.1,.1,.1,.1,.1,.2,.7]]

args["step_lrs"] = [1e-4,5e-5,1e-5,1e-5,5e-6,5e-6,5e-6]
args["step_lrs"] = [i*1e-1 for i in args["step_lrs"]]

inputs, extra_in, outputs = gen_data_025_lateral_3D(inputs_str,extra_in_str,outputs_str,depth_list,lag,lat,lon,Nb)


data_in_test = gen_data_in_test(0,e_test,N_test,lag,hist,inputs,extra_in)

data_out_test = gen_data_out_test(0,e_test,N_test,lag,hist,outputs)

test_data = data_CNN_Lateral(data_in_test,data_out_test,wet_list,N_atm,Nb,device="cpu",norms=norm_vals,N_vars=N_var) 

time_test = time_vec[e_test:(e_test+lag*N_test)]

mean_out = test_data.norm_vals['m_out']  
std_out = test_data.norm_vals['s_out'] 
mean_in = test_data.norm_vals['m_in']  
std_in = test_data.norm_vals['s_in'] 

N = 5

plt.style.use('bmh')

clist_1 = [cmocean.cm.thermal(i/(N-.5)) for i in range(1,N)]
clist_2 = ['#d7191c','#abd9e9','#2c7bb6','#fdae61']
clist_3 = ["#91B59A","#D6A922","#1E88E5","#A00B41"]
clist_5 = ["#A00B41","#00DCDE","#A6BD00","#3300EA"]
clist_6 = ["#A00B41","#DE7400","#00BD8E","#3300EA"]
clist = clist_5

#model_pred = xr.open_zarr(f"{os.environ['SCRATCH']}/Emulation/Preds/Pred_UNet_3D_"+region+"_in_"+str_in+"ext_"+str_ext+"N_test_"+str(N_test)+".zarr").to_array().to_numpy().squeeze()
model_pred_1 = xr.open_zarr(f"{os.environ['SCRATCH']}/Emulation/Preds/Pred_UNet_3D_Gulf_Stream_Ext_in_um_vm_Tm_ext_tau_u_tau_v_t_ref_N_test_400hist_0lag_1step_4.zarr").to_array().to_numpy().squeeze()
model_pred_2 = xr.open_zarr(f"{os.environ['SCRATCH']}/Emulation/Preds/Pred_UNet_3D_Gulf_Stream_Ext_in_um_vm_Tm_ext_tau_u_tau_v_t_ref_N_test_400hist_0lag_2step_4.zarr").to_array().to_numpy().squeeze()
model_pred_5 = xr.open_zarr(f"{os.environ['SCRATCH']}/Emulation/Preds/Pred_UNet_3D_Gulf_Stream_Ext_in_um_vm_Tm_ext_tau_u_tau_v_t_ref_N_test_400hist_0lag_5step_4.zarr").to_array().to_numpy().squeeze()
model_pred_8 = xr.open_zarr(f"{os.environ['SCRATCH']}/Emulation/Preds/Pred_UNet_3D_Gulf_Stream_Ext_in_um_vm_Tm_ext_tau_u_tau_v_t_ref_N_test_400hist_0lag_1step_8.zarr").to_array().to_numpy().squeeze()

#print("model test_data type  = ",type(test_data))
#print("model test_data type 0,3 = ",type(test_data[:,:,:,0:3]))
#print("model test_data shape  = ",test_data.shape)
#print("wet shape =",wet.shape)
'''
N_plot = 200
KE_spec_1, KE_spec_true = gen_KE_spectrum(N_plot,test_data,model_pred_1,grids,wet, depth=[0,3])
KE_spec_2, KE_spec_true = gen_KE_spectrum(int(N_plot/2),test_data,model_pred_2,grids,wet, depth=[0,3])
KE_spec_5, KE_spec_true = gen_KE_spectrum(int(N_plot/5),test_data,model_pred_5,grids,wet, depth=[0,3])
KE_spec_8, KE_spec_true = gen_KE_spectrum(N_plot,test_data,model_pred_8,grids,wet, depth=[0,3])


plt.loglog(KE_spec_1.freq_r,KE_spec_1,c=clist[0],label = r"$\Delta t = 1$")
plt.loglog(KE_spec_2.freq_r,KE_spec_2,c=clist[1],label = r"$\Delta t = 2$")
plt.loglog(KE_spec_5.freq_r,KE_spec_5,c=clist[2],label = r"$\Delta t = 5$")
plt.loglog(KE_spec_8.freq_r,KE_spec_8,c=clist[3],label = r"$\Delta t = 1,~ N = 8$")

plt.loglog(KE_spec_true.freq_r,KE_spec_true,"--k")

plt.legend(loc= "lower left")
plt.title('KE Spec at depth [0,3]')
plt.savefig('plots/unet-3d/ke-spec_0_3.png')
plt.clf()
print('finish ke spec 03')

N_plot = 200
KE_spec_1, KE_spec_true = gen_KE_spectrum(N_plot,test_data,model_pred_1,grids,wet, depth=[3,6])
KE_spec_2, KE_spec_true = gen_KE_spectrum(int(N_plot/2),test_data,model_pred_2,grids,wet, depth=[3,6])
KE_spec_5, KE_spec_true = gen_KE_spectrum(int(N_plot/5),test_data,model_pred_5,grids,wet, depth=[3,6])
KE_spec_8, KE_spec_true = gen_KE_spectrum(N_plot,test_data,model_pred_8,grids,wet, depth=[3,6])


plt.loglog(KE_spec_1.freq_r,KE_spec_1,c=clist[0],label = r"$\Delta t = 1$")
plt.loglog(KE_spec_2.freq_r,KE_spec_2,c=clist[1],label = r"$\Delta t = 2$")
plt.loglog(KE_spec_5.freq_r,KE_spec_5,c=clist[2],label = r"$\Delta t = 5$")
plt.loglog(KE_spec_8.freq_r,KE_spec_8,c=clist[3],label = r"$\Delta t = 1,~ N = 8$")

plt.loglog(KE_spec_true.freq_r,KE_spec_true,"--k")

plt.legend(loc= "lower left")
plt.title('KE Spec at depth [3,6]')
plt.savefig('plots/unet-3d/ke-spec_3_6.png')
plt.clf()
print('finish ke spec 36')

N_plot = 200
KE_spec_1, KE_spec_true = gen_KE_spectrum(N_plot,test_data,model_pred_1,grids,wet, depth=[6,9])
KE_spec_2, KE_spec_true = gen_KE_spectrum(int(N_plot/2),test_data,model_pred_2,grids,wet, depth=[6,9])
KE_spec_5, KE_spec_true = gen_KE_spectrum(int(N_plot/5),test_data,model_pred_5,grids,wet, depth=[6,9])
KE_spec_8, KE_spec_true = gen_KE_spectrum(N_plot,test_data,model_pred_8,grids,wet, depth=[6,9])


plt.loglog(KE_spec_1.freq_r,KE_spec_1,c=clist[0],label = r"$\Delta t = 1$")
plt.loglog(KE_spec_2.freq_r,KE_spec_2,c=clist[1],label = r"$\Delta t = 2$")
plt.loglog(KE_spec_5.freq_r,KE_spec_5,c=clist[2],label = r"$\Delta t = 5$")
plt.loglog(KE_spec_8.freq_r,KE_spec_8,c=clist[3],label = r"$\Delta t = 1,~ N = 8$")

plt.loglog(KE_spec_true.freq_r,KE_spec_true,"--k")

plt.legend(loc= "lower left")
plt.title('KE Spec at depth [6,9]')
plt.savefig('plots/unet-3d/ke-spec_6_9.png')
plt.clf()
print('finish ke spec 69')

	
N_plot = 200

KE_1, KE_true = compute_KE(N_plot,test_data,model_pred_1,area,wet_bool, depth=[0,3])
KE_2, KE_true = compute_KE(int(N_plot/2),test_data,model_pred_2,area,wet_bool, depth=[0,3])
KE_5, KE_true = compute_KE(int(N_plot/5),test_data,model_pred_5,area,wet_bool, depth=[0,3])
KE_8, KE_true = compute_KE(N_plot,test_data,model_pred_8,area,wet_bool, depth=[0,3])

rho = 1020

plt.plot(np.arange(1,N_plot+1),KE_1*rho,c=clist[0],label = r"$\Delta t = 1$")
plt.plot(np.arange(2,N_plot+1,2),KE_2*rho,c=clist[1],label = r"$\Delta t = 2$")
plt.plot(np.arange(5,N_plot+1,5),KE_5*rho,c=clist[2],label = r"$\Delta t = 5$")
plt.plot(np.arange(1,N_plot+1),KE_8*rho,c=clist[3],label = r"$\Delta t = 1,~ N = 8$")

plt.plot(np.arange(1,N_plot+1),KE_true*rho,"--k")
plt.xlabel("time (days)")
plt.ylabel("Kinetic Energy")
plt.title("KE vs Time at depth [0,3]")
plt.legend(loc= "lower left")
plt.savefig('plots/unet-3d/ke-time_0_3.png')
plt.clf()
print('finish ke t 03')

N_plot = 200

KE_1, KE_true = compute_KE(N_plot,test_data,model_pred_1,area,wet_bool, depth=[3,6])
KE_2, KE_true = compute_KE(int(N_plot/2),test_data,model_pred_2,area,wet_bool, depth=[3,6])
KE_5, KE_true = compute_KE(int(N_plot/5),test_data,model_pred_5,area,wet_bool, depth=[3,6])
KE_8, KE_true = compute_KE(N_plot,test_data,model_pred_8,area,wet_bool, depth=[3,6])

rho = 1020

plt.plot(np.arange(1,N_plot+1),KE_1*rho,c=clist[0],label = r"$\Delta t = 1$")
plt.plot(np.arange(2,N_plot+1,2),KE_2*rho,c=clist[1],label = r"$\Delta t = 2$")
plt.plot(np.arange(5,N_plot+1,5),KE_5*rho,c=clist[2],label = r"$\Delta t = 5$")
plt.plot(np.arange(1,N_plot+1),KE_8*rho,c=clist[3],label = r"$\Delta t = 1,~ N = 8$")

plt.plot(np.arange(1,N_plot+1),KE_true*rho,"--k")
plt.xlabel("time (days)")
plt.ylabel("Kinetic Energy")
plt.title("KE vs Time at depth [3,6]")
plt.legend(loc= "lower left")
plt.savefig('plots/unet-3d/ke-time_3_6.png')
plt.clf()
print('finish ke t 36')

N_plot = 200

KE_1, KE_true = compute_KE(N_plot,test_data,model_pred_1,area,wet_bool, depth=[6,9])
KE_2, KE_true = compute_KE(int(N_plot/2),test_data,model_pred_2,area,wet_bool, depth=[6,9])
KE_5, KE_true = compute_KE(int(N_plot/5),test_data,model_pred_5,area,wet_bool, depth=[6,9])
KE_8, KE_true = compute_KE(N_plot,test_data,model_pred_8,area,wet_bool, depth=[6,9])

rho = 1020

plt.plot(np.arange(1,N_plot+1),KE_1*rho,c=clist[0],label = r"$\Delta t = 1$")
plt.plot(np.arange(2,N_plot+1,2),KE_2*rho,c=clist[1],label = r"$\Delta t = 2$")
plt.plot(np.arange(5,N_plot+1,5),KE_5*rho,c=clist[2],label = r"$\Delta t = 5$")
plt.plot(np.arange(1,N_plot+1),KE_8*rho,c=clist[3],label = r"$\Delta t = 1,~ N = 8$")

plt.plot(np.arange(1,N_plot+1),KE_true*rho,"--k")
plt.xlabel("time (days)")
plt.ylabel("Kinetic Energy")
plt.title("KE vs Time at depth [6,9]")
plt.legend(loc= "lower left")
plt.savefig('plots/unet-3d/ke-time_6_9.png')
plt.clf()
print('finish ke t 69')

def big_plot(true_KE, pred_KE_1, pred_KE_2, pred_KE_5, pred_KE_8, depth = []):
    plt.rcParams.update({'font.size': 15})
    
    fig, axs = plt.subplots(2, 3, figsize=(12,8),
                    gridspec_kw={'width_ratios': [1,1,1], 'height_ratios': [1,1], 'wspace': 0.25,'hspace':.5},
                    subplot_kw={'projection': ccrs.PlateCarree()})

    vmin = 0
    vmax =250
    
    x_plot = grids["x_C"][Nb:-Nb,Nb:-Nb]
    y_plot = grids["y_C"][Nb:-Nb,Nb:-Nb]
    
    cmap = cmocean.cm.thermal
    

    plt1 = axs[0,0].pcolormesh(x_plot, y_plot,
                    true_KE[Nb:-Nb,Nb:-Nb]*wet_nan[Nb:-Nb,Nb:-Nb],
                    cmap=cmap,vmin=vmin,vmax=vmax,shading='auto')
    
    axs[0,0].add_feature(cart.feature.LAND, zorder=100, edgecolor='k')
    gl = axs[0,0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
                      
    gl.top_labels = False
    gl.right_labels = False
    gl.yrotation = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    axs[0,0].set_title(r"CM2.6",size = 15)
    
    pos = axs[1, 0].get_position()
    
    # Set the new anchor point to be in the middle
    new_pos = [pos.x0-.1, pos.y0+.15, pos.width*1.25, pos.height*1.5]  # Adjust 0.2 as needed
    
    # Create a new axes with the adjusted position
    cax = fig.add_axes(new_pos)
    
    
    cbar = plt.colorbar(plt1, ax=cax, orientation="horizontal",aspect=10)
    cbar.ax.tick_params(labelsize=16)  # Set the font size for tick labels
    
    cbar.set_ticks([vmin,0, vmax])  
        
    cbar.set_label(r"KE $\left( \frac{J}{m^2} \right)$", fontsize=20)
    
    fig.delaxes(axs[1,0])
    fig.delaxes(cax)
    
    plt1 = axs[0,1].pcolormesh(x_plot, y_plot,
                    pred_KE_1[Nb:-Nb,Nb:-Nb]*wet_nan[Nb:-Nb,Nb:-Nb],
                    cmap=cmap,vmin=vmin,vmax=vmax,shading='auto')
    
    axs[0,1].add_feature(cart.feature.LAND, zorder=100, edgecolor='k')
    gl = axs[0,1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.yrotation = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    axs[0,1].set_title(r"$\Delta t = 1$",size = 15)
    
    
    plt1 = axs[0,2].pcolormesh(x_plot, y_plot,
                    pred_KE_2[Nb:-Nb,Nb:-Nb]*wet_nan[Nb:-Nb,Nb:-Nb],
                    cmap=cmap,vmin=vmin,vmax=vmax,shading='auto')
    
    axs[0,2].add_feature(cart.feature.LAND, zorder=100, edgecolor='k')
    gl = axs[0,2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.yrotation = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    axs[0,2].set_title(r"$\Delta t = 2$",size = 15)
    
    
    
    
    plt1 = axs[1,1].pcolormesh(x_plot, y_plot,
                          pred_KE_5[Nb:-Nb,Nb:-Nb]*wet_nan[Nb:-Nb,Nb:-Nb],
                      cmap=cmap,vmin=vmin,vmax=vmax,shading='auto')
    
    axs[1,1].add_feature(cart.feature.LAND, zorder=100, edgecolor='k')
    gl = axs[1,1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.yrotation = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    axs[1,1].set_title(r"$\Delta t = 5$",size = 15)
    
    
    
    plt1 = axs[1,2].pcolormesh(x_plot, y_plot,
                          pred_KE_8[Nb:-Nb,Nb:-Nb]*wet_nan[Nb:-Nb,Nb:-Nb],
                      cmap=cmap,vmin=vmin,vmax=vmax,shading='auto')
    
    axs[1,2].add_feature(cart.feature.LAND, zorder=100, edgecolor='k')
    gl = axs[1,2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.yrotation = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    axs[1,2].set_title(r"$\Delta t = 1, ~ N = 8$",size = 15)
    
    
    
    region_title = ""
    
    for i in region:
        if region == "Quiescent_Ext":
            region_title = "South Pacific"
        elif region == "Africa_Ext":
            region_title = "African Cape"         
        elif i == "_":
            region_title += " "
        elif i == "E":
            break
        else:
            region_title+= i
    region_title = str(region_title)
    plt.title(f'big plot {"_".join([str(x) for x in depth])}')
    plt.savefig(f'plots/unet-3d/big-plot-{"_".join([str(x) for x in depth])}.png')
    plt.clf()
    print(f'finish big-plot-{"_".join([str(x) for x in depth])}')
    

pred_KE_1, true_KE = gen_KE(1000,test_data,model_pred_1, depth=[0,3])
pred_KE_1 = pred_KE_1.mean(0)
pred_KE_2, true_KE = gen_KE(1000,test_data,model_pred_2, depth=[0,3])
pred_KE_2 = pred_KE_2.mean(0)
pred_KE_5, true_KE = gen_KE(1000,test_data,model_pred_5, depth=[0,3])
pred_KE_5 = pred_KE_5.mean(0)
pred_KE_8, true_KE = gen_KE(1000,test_data,model_pred_8, depth=[0,3])
pred_KE_8 = pred_KE_8.mean(0)
true_KE = true_KE.mean(0)

big_plot(true_KE, pred_KE_1, pred_KE_2, pred_KE_5, pred_KE_8, depth=[0,3])

pred_KE_1, true_KE = gen_KE(1000,test_data,model_pred_1, depth=[3,6])
pred_KE_1 = pred_KE_1.mean(0)
pred_KE_2, true_KE = gen_KE(1000,test_data,model_pred_2, depth=[3,6])
pred_KE_2 = pred_KE_2.mean(0)
pred_KE_5, true_KE = gen_KE(1000,test_data,model_pred_5, depth=[3,6])
pred_KE_5 = pred_KE_5.mean(0)
pred_KE_8, true_KE = gen_KE(1000,test_data,model_pred_8, depth=[3,6])
pred_KE_8 = pred_KE_8.mean(0)
true_KE = true_KE.mean(0)

big_plot(true_KE, pred_KE_1, pred_KE_2, pred_KE_5, pred_KE_8, depth=[3,6])

pred_KE_1, true_KE = gen_KE(1000,test_data,model_pred_1, depth=[6,9])
pred_KE_1 = pred_KE_1.mean(0)
pred_KE_2, true_KE = gen_KE(1000,test_data,model_pred_2, depth=[6,9])
pred_KE_2 = pred_KE_2.mean(0)
pred_KE_5, true_KE = gen_KE(1000,test_data,model_pred_5, depth=[6,9])
pred_KE_5 = pred_KE_5.mean(0)
pred_KE_8, true_KE = gen_KE(1000,test_data,model_pred_8, depth=[6,9])
pred_KE_8 = pred_KE_8.mean(0)
true_KE = true_KE.mean(0)

big_plot(true_KE, pred_KE_1, pred_KE_2, pred_KE_5, pred_KE_8, depth=[6,9])

N_plot = 200

enst_spec_1, enst_spec_true = gen_enstrophy_spectrum(N_plot,test_data,model_pred_1,grids,wet,wet_lap, depth=[0,3])
enst_spec_2, enst_spec_true = gen_enstrophy_spectrum(int(N_plot/2),test_data,model_pred_2,grids,wet,wet_lap, depth=[0,3])
enst_spec_5, enst_spec_true = gen_enstrophy_spectrum(int(N_plot/5),test_data,model_pred_5,grids,wet,wet_lap, depth=[0,3])
enst_spec_8, enst_spec_true = gen_enstrophy_spectrum(N_plot,test_data,model_pred_8,grids,wet,wet_lap, depth=[0,3])

plt.loglog(enst_spec_1.freq_r,enst_spec_1,c=clist[0],label = r"$\Delta t = 1$")
plt.loglog(enst_spec_2.freq_r,enst_spec_2,c=clist[1],label = r"$\Delta t = 2$")
plt.loglog(enst_spec_5.freq_r,enst_spec_5,c=clist[2],label = r"$\Delta t = 5$")
plt.loglog(enst_spec_8.freq_r,enst_spec_8,c=clist[3],label = r"$\Delta t = 1,~ N = 8$")

plt.loglog(enst_spec_true.freq_r,enst_spec_true,"--k")

plt.title("Enstrophy Spec at depth [0,3]")
plt.legend(loc= "lower left")
plt.savefig('plots/unet-3d/ens-spec_0_3.png')
plt.clf()
print('finish ens-spec_0_3')

N_plot = 200

enst_spec_1, enst_spec_true = gen_enstrophy_spectrum(N_plot,test_data,model_pred_1,grids,wet,wet_lap, depth=[3,6])
enst_spec_2, enst_spec_true = gen_enstrophy_spectrum(int(N_plot/2),test_data,model_pred_2,grids,wet,wet_lap, depth=[3,6])
enst_spec_5, enst_spec_true = gen_enstrophy_spectrum(int(N_plot/5),test_data,model_pred_5,grids,wet,wet_lap, depth=[3,6])
enst_spec_8, enst_spec_true = gen_enstrophy_spectrum(N_plot,test_data,model_pred_8,grids,wet,wet_lap, depth=[3,6])

plt.loglog(enst_spec_1.freq_r,enst_spec_1,c=clist[0],label = r"$\Delta t = 1$")
plt.loglog(enst_spec_2.freq_r,enst_spec_2,c=clist[1],label = r"$\Delta t = 2$")
plt.loglog(enst_spec_5.freq_r,enst_spec_5,c=clist[2],label = r"$\Delta t = 5$")
plt.loglog(enst_spec_8.freq_r,enst_spec_8,c=clist[3],label = r"$\Delta t = 1,~ N = 8$")

plt.loglog(enst_spec_true.freq_r,enst_spec_true,"--k")

plt.title("Enstrophy Spec at depth [3,6]")
plt.legend(loc= "lower left")
plt.savefig('plots/unet-3d/ens-spec_3_6.png')
plt.clf()
print('finish ens-spec_3_6')

N_plot = 200

enst_spec_1, enst_spec_true = gen_enstrophy_spectrum(N_plot,test_data,model_pred_1,grids,wet,wet_lap, depth=[6,9])
enst_spec_2, enst_spec_true = gen_enstrophy_spectrum(int(N_plot/2),test_data,model_pred_2,grids,wet,wet_lap, depth=[6,9])
enst_spec_5, enst_spec_true = gen_enstrophy_spectrum(int(N_plot/5),test_data,model_pred_5,grids,wet,wet_lap, depth=[6,9])
enst_spec_8, enst_spec_true = gen_enstrophy_spectrum(N_plot,test_data,model_pred_8,grids,wet,wet_lap, depth=[6,9])

plt.loglog(enst_spec_1.freq_r,enst_spec_1,c=clist[0],label = r"$\Delta t = 1$")
plt.loglog(enst_spec_2.freq_r,enst_spec_2,c=clist[1],label = r"$\Delta t = 2$")
plt.loglog(enst_spec_5.freq_r,enst_spec_5,c=clist[2],label = r"$\Delta t = 5$")
plt.loglog(enst_spec_8.freq_r,enst_spec_8,c=clist[3],label = r"$\Delta t = 1,~ N = 8$")

plt.loglog(enst_spec_true.freq_r,enst_spec_true,"--k")

plt.title("Enstrophy Spec at depth [6,9]")
plt.legend(loc= "lower left")
plt.savefig('plots/unet-3d/ens-spec_6_9.png')
plt.clf()
print('finish ens-spec_6_9')

N_plot = 200

enst_1, enst_true = gen_enstrophy(N_plot,test_data,model_pred_1,dx, dy, Nb, wet_lap, depth=[0,3])
enst_1 = enst_1.mean(axis=(1,2))
enst_2, enst_true = gen_enstrophy(int(N_plot/2),test_data,model_pred_2,dx, dy, Nb, wet_lap, depth=[0,3])
enst_2 = enst_2.mean(axis=(1,2))
enst_5, enst_true = gen_enstrophy(int(N_plot/5),test_data,model_pred_5,dx, dy, Nb, wet_lap, depth=[0,3])
enst_5 = enst_5.mean(axis=(1,2))
enst_8, enst_true = gen_enstrophy(N_plot,test_data,model_pred_8,dx, dy, Nb, wet_lap, depth=[0,3])
enst_8 = enst_8.mean(axis=(1,2))
enst_true = enst_true.mean(axis=(1,2))

plt.plot(np.arange(1,N_plot+1),enst_1,c=clist[0],label = r"$\Delta t = 1$")
plt.plot(np.arange(2,N_plot+1,2),enst_2,c=clist[1],label = r"$\Delta t = 2$")
plt.plot(np.arange(5,N_plot+1,5),enst_5,c=clist[2],label = r"$\Delta t = 5$")
plt.plot(np.arange(1,N_plot+1),enst_8,c=clist[3],label = r"$\Delta t = 1,~ N = 8$")

plt.plot(np.arange(1,N_plot+1),enst_true,"--k")
plt.xlabel("time (days)")
plt.ylabel("Enstrophy")
plt.title("Enstrophy vs Time at depth [0,3]")
plt.legend(loc= "lower left")
plt.savefig('plots/unet-3d/ens-time_0_3.png')
plt.clf()
print('finish ens-time_0_3')

N_plot = 200

enst_1, enst_true = gen_enstrophy(N_plot,test_data,model_pred_1,dx, dy, Nb, wet_lap, depth=[3,6])
enst_1 = enst_1.mean(axis=(1,2))
enst_2, enst_true = gen_enstrophy(int(N_plot/2),test_data,model_pred_2,dx, dy, Nb, wet_lap, depth=[3,6])
enst_2 = enst_2.mean(axis=(1,2))
enst_5, enst_true = gen_enstrophy(int(N_plot/5),test_data,model_pred_5,dx, dy, Nb, wet_lap, depth=[3,6])
enst_5 = enst_5.mean(axis=(1,2))
enst_8, enst_true = gen_enstrophy(N_plot,test_data,model_pred_8,dx, dy, Nb, wet_lap, depth=[3,6])
enst_8 = enst_8.mean(axis=(1,2))
enst_true = enst_true.mean(axis=(1,2))

plt.plot(np.arange(1,N_plot+1),enst_1,c=clist[0],label = r"$\Delta t = 1$")
plt.plot(np.arange(2,N_plot+1,2),enst_2,c=clist[1],label = r"$\Delta t = 2$")
plt.plot(np.arange(5,N_plot+1,5),enst_5,c=clist[2],label = r"$\Delta t = 5$")
plt.plot(np.arange(1,N_plot+1),enst_8,c=clist[3],label = r"$\Delta t = 1,~ N = 8$")

plt.plot(np.arange(1,N_plot+1),enst_true,"--k")
plt.xlabel("time (days)")
plt.ylabel("Enstrophy")
plt.title("Enstrophy vs Time at depth [3,6]")
plt.legend(loc= "lower left")
plt.savefig('plots/unet-3d/ens-time_3_6.png')
plt.clf()
print('finish ens-time_3_6')

N_plot = 200

enst_1, enst_true = gen_enstrophy(N_plot,test_data,model_pred_1,dx, dy, Nb, wet_lap, depth=[6,9])
enst_1 = enst_1.mean(axis=(1,2))
enst_2, enst_true = gen_enstrophy(int(N_plot/2),test_data,model_pred_2,dx, dy, Nb, wet_lap, depth=[6,9])
enst_2 = enst_2.mean(axis=(1,2))
enst_5, enst_true = gen_enstrophy(int(N_plot/5),test_data,model_pred_5,dx, dy, Nb, wet_lap, depth=[6,9])
enst_5 = enst_5.mean(axis=(1,2))
enst_8, enst_true = gen_enstrophy(N_plot,test_data,model_pred_8,dx, dy, Nb, wet_lap, depth=[6,9])
enst_8 = enst_8.mean(axis=(1,2))
enst_true = enst_true.mean(axis=(1,2))

plt.plot(np.arange(1,N_plot+1),enst_1,c=clist[0],label = r"$\Delta t = 1$")
plt.plot(np.arange(2,N_plot+1,2),enst_2,c=clist[1],label = r"$\Delta t = 2$")
plt.plot(np.arange(5,N_plot+1,5),enst_5,c=clist[2],label = r"$\Delta t = 5$")
plt.plot(np.arange(1,N_plot+1),enst_8,c=clist[3],label = r"$\Delta t = 1,~ N = 8$")

plt.plot(np.arange(1,N_plot+1),enst_true,"--k")
plt.xlabel("time (days)")
plt.ylabel("Enstrophy")
plt.title("Enstrophy vs Time at depth [6,9]")
plt.legend(loc= "lower left")
plt.savefig('plots/unet-3d/ens-time_6_9.png')
plt.clf()
print('finish ens-time_6_9')

def spatial_matching_stats(depth=[]):
    start = 0
    if len(depth) != 0:
        start, end = depth
    u_test = np.array(test_data[:][1][:,start]*std_out[start] +mean_out[start])
    v_test = np.array(test_data[:][1][:,start+1]*std_out[start+1] +mean_out[start+1])
    T_test = np.array(test_data[:][1][:,start+2]*std_out[start+2] +mean_out[start+2])
    
    N_eval = 200
    corr_T_1, corr_T_true = compute_corrs_single(N_eval, T_test, model_pred_1[:,:,:,start+2],area, wet_bool,std_out[start+2],mean_out[start+2])
    corr_T_2, corr_T_true = compute_corrs_single(int(N_eval/2), T_test[::2], model_pred_2[:,:,:,start+2],area, wet_bool,std_out[start+2],mean_out[start+2])
    corr_T_5, corr_T_true = compute_corrs_single(int(N_eval/5), T_test[::5], model_pred_5[:,:,:,start+2],area, wet_bool,std_out[start+2],mean_out[start+2])
    corr_T_8, corr_T_true = compute_corrs_single(N_eval, T_test, model_pred_8[:,:,:,start+2],area, wet_bool,std_out[start+2],mean_out[start+2])
    
    plt.plot(np.arange(1,N_eval+1),corr_T_1,c=clist[0],label = r"$\Delta t = 1$")
    plt.plot(np.arange(2,N_eval+1,2),corr_T_2,c=clist[1],label = r"$\Delta t = 2$")
    plt.plot(np.arange(5,N_eval+1,5),corr_T_5,c=clist[2],label = r"$\Delta t = 5$")
    plt.plot(np.arange(1,N_eval+1),corr_T_8,c=clist[3],label = r"$\Delta t = 1,~ N = 8$")
    
    plt.plot(np.arange(1,N_eval+1),corr_T_true,"--k")
    plt.xlabel("time (days)")
    plt.ylabel(r"Correlation $T$")
    plt.ylim([0,1])
    plt.xlim([0,N_eval])
    
    plt.legend(loc= "lower left")
    plt.title(f'Correlation T vs time {"_".join([str(x) for x in depth])}')
    plt.savefig(f'plots/unet-3d/corrT-time-{"_".join([str(x) for x in depth])}.png')
    plt.clf()
    print(f'finish corrT-time-{"_".join([str(x) for x in depth])}')

spatial_matching_stats([0,3])
spatial_matching_stats([3,6])
spatial_matching_stats([6,9])

def accTvsTime(depth=[]):
    start = 0
    if len(depth) != 0:
        start, end = depth
    N_eval = 73
    u_test = np.array(test_data[:][1][:,start]*std_out[start] +mean_out[start])
    v_test = np.array(test_data[:][1][:,start+1]*std_out[start+1] +mean_out[start+1])
    T_test = np.array(test_data[:][1][:,start+2]*std_out[start+2] +mean_out[start+2])
    ACC_T_1, ACC_T_true = compute_ACC_single(N_eval, T_test, model_pred_1[:,:,:,start+2],
                                             clim[:,:,:,start+2],time_test,area, wet_bool)
    ACC_T_2, ACC_T_true = compute_ACC_single(int(N_eval/2), T_test[::2], model_pred_2[:,:,:,start+2],
                                             clim[:,:,:,start+2],time_test,area, wet_bool)
    ACC_T_5, ACC_T_true = compute_ACC_single(int(N_eval/5), T_test[::5], model_pred_5[:,:,:,start+2],
                                             clim[:,:,:,start+2],time_test,area, wet_bool)
    ACC_T_8, ACC_T_true = compute_ACC_single(N_eval, T_test, model_pred_8[:,:,:,start+2],
                                             clim[:,:,:,start+2],time_test,area, wet_bool)
                                             
    plt.plot(np.arange(1,N_eval+1),ACC_T_1,c=clist[0],label = r"$\Delta t = 1$")
    plt.plot(np.arange(2,N_eval+1,2),ACC_T_2,c=clist[1],label = r"$\Delta t = 2$")
    plt.plot(np.arange(5,N_eval+1,5),ACC_T_5,c=clist[2],label = r"$\Delta t = 5$")
    plt.plot(np.arange(1,N_eval+1),ACC_T_8,c=clist[3],label = r"$\Delta t = 1,~ N = 8$")
    
    plt.plot(np.arange(1,N_eval+1),ACC_T_true,"--k")
    plt.xlabel("time (days)")
    plt.ylabel(r"ACC $T$")
    plt.ylim([0,1])
    plt.xlim([0,N_eval])
    
    plt.legend(loc= "lower left")
    plt.title(f'ACC T vs time {"_".join([str(x) for x in depth])}')
    plt.savefig(f'plots/unet-3d/accT-time-{"_".join([str(x) for x in depth])}.png')
    plt.clf()
    print(f'finish accT-time-{"_".join([str(x) for x in depth])}')
    
accTvsTime([0,3])
accTvsTime([3,6])
accTvsTime([6,9])

def rmseTvsTime(depth=[]):
    start = 0
    if len(depth) != 0:
        start, end = depth
    N_eval = 200
    u_test = np.array(test_data[:][1][:,start]*std_out[start] +mean_out[start])
    v_test = np.array(test_data[:][1][:,start+1]*std_out[start+1] +mean_out[start+1])
    T_test = np.array(test_data[:][1][:,start+2]*std_out[start+2] +mean_out[start+2])
    RMSE_T_1, RMSE_T_true = compute_RMSE_single(N_eval, T_test, model_pred_1[:,:,:,start+2],
                                             area, wet_bool)
    RMSE_T_2, RMSE_T_true = compute_RMSE_single(int(N_eval/2), T_test[::2], model_pred_2[:,:,:,start+2],
                                             area, wet_bool)
    RMSE_T_5, RMSE_T_true = compute_RMSE_single(int(N_eval/5), T_test[::5], model_pred_5[:,:,:,start+2],
                                             area, wet_bool)
    RMSE_T_8, RMSE_T_true = compute_RMSE_single(N_eval, T_test, model_pred_8[:,:,:,start+2],
                                             area, wet_bool)
                                             
    plt.plot(np.arange(1,N_eval+1),RMSE_T_1,c=clist[0],label = r"$\Delta t = 1$")
    plt.plot(np.arange(2,N_eval+1,2),RMSE_T_2,c=clist[1],label = r"$\Delta t = 2$")
    plt.plot(np.arange(5,N_eval+1,5),RMSE_T_5,c=clist[2],label = r"$\Delta t = 5$")
    plt.plot(np.arange(1,N_eval+1),RMSE_T_8,c=clist[3],label = r"$\Delta t = 1,~ N = 8$")
    
    plt.plot(np.arange(1,N_eval+1),RMSE_T_true,"--k")
    plt.xlabel("time (days)")
    plt.ylabel(r"RMSE $T$")
    plt.xlim([0,N_eval])
    
    plt.legend(loc= "lower left")
    plt.title(f'RMSE T vs time {"_".join([str(x) for x in depth])}')
    plt.savefig(f'plots/unet-3d/rmseT-time-{"_".join([str(x) for x in depth])}.png')
    plt.clf()
    print(f'finish rmseT-time-{"_".join([str(x) for x in depth])}')
rmseTvsTime([0,3])
rmseTvsTime([3,6])
rmseTvsTime([6,9])
'''
def pdfTvsT(depth=[]):
    start = 0
    if len(depth) != 0:
        start, end = depth
    #final 100 days of rollout
    var_list = {"1":r"$\bar{v}$ (m/s)","0":r"$\bar{u}$ (m/s)","2":r"$\bar{T}$ ~ $(^\circ C)$"}
    
    ind_plot = start+2
    N_days = 50
    day_start = 10
    
    true_field = (test_data[day_start:day_start+N_days][1][:,ind_plot,wet_bool].flatten()*std_out[ind_plot])+mean_out[ind_plot]
    true_field = np.array(true_field)
    #print(true_field.shape)
    field_1 = (model_pred_1[day_start:day_start+N_days,wet_bool,ind_plot].flatten())
    day_2 = int(day_start/2)
    field_2 = (model_pred_2[day_2:day_2+int(N_days/2),wet_bool,ind_plot].flatten())
    day_5 = int(day_start/5)
    field_5 = (model_pred_5[day_5:day_5+int(N_days/5),wet_bool,ind_plot].flatten())
    field_8 = (model_pred_8[day_start:day_start+N_days,wet_bool,ind_plot].flatten())
    
    true_pdf, bins_true = np.histogram(true_field,bins = 150,density = True);
    bins_true = (bins_true[1:]+bins_true[:-1])/2
    pdf_1, bins_1 = np.histogram(field_1,bins = 150,density = True);
    bins_1 = (bins_1[1:]+bins_1[:-1])/2
    pdf_2, bins_2 = np.histogram(field_2,bins = 150,density = True);
    bins_2 = (bins_2[1:]+bins_2[:-1])/2
    pdf_5, bins_5 = np.histogram(field_5,bins = 150,density = True);
    bins_5 = (bins_5[1:]+bins_5[:-1])/2
    pdf_8, bins_8 = np.histogram(field_8,bins = 150,density = True);
    bins_8 = (bins_8[1:]+bins_8[:-1])/2
    
    
    plt.semilogy(bins_true,true_pdf,lw=2,c="k",label = r"CM 2.6",zorder=10)
    plt.semilogy(bins_1,pdf_1,lw=2,color=clist[0],label = r"$\Delta t = 1$")
    plt.semilogy(bins_2,pdf_2,lw=2,color=clist[1],label = r"$\Delta t = 2$")
    plt.semilogy(bins_5,pdf_5,lw=2,color=clist[2],label = r"$\Delta t = 5$")
    plt.semilogy(bins_8,pdf_8,lw=2,color=clist[3],label = r"$\Delta t = 1,~ N = 8$")
    #print([true_pdf.min()*2,true_pdf.max()*2.5])
    plt.ylim([true_pdf.min()*2,true_pdf.max()*2.5])
    
    plt.legend(loc="lower center",fontsize=12)
    
    plt.xlabel(var_list[str(ind_plot%3)])
    plt.ylabel(r"${p(}$" + var_list[str(ind_plot%3)][:9]+"${)}$")
    plt.title(f'PDF T vs T {"_".join([str(x) for x in depth])}')
    plt.savefig(f'plots/unet-3d/pdfT-t-{"_".join([str(x) for x in depth])}.png')
    plt.clf()
    print(f'finish pdfT-t-{"_".join([str(x) for x in depth])}')
pdfTvsT([0,3])
pdfTvsT([3,6])
pdfTvsT([6,9])
