# %%
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

import cartopy.crs as ccrs
import cartopy as cart
# import cmocean
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
from Eval_Funcs import *
from Parallel import *
import numpy.fft as fft
import sys


# %%
exp_num_in = "3"
exp_num_extra = "12"
exp_num_res = "4"


mse = torch.nn.MSELoss()

region = "Gulf_Stream"    
network = "U_net"

interval = 3

N_samples = 1000
N_val = 400
N_test = 1000

factor = 10

hist = 1

lag = 1

steps = 1

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

    
if region == "Kuroshio":
    lat = [15,41]
    lon = [-215, -185]
elif region == "Gulf_Stream":
    lat = [25, 50]
    lon = [-70,-35]
elif region == "Tropics":
    lat = [-5,25]
    lon = [-95,-65]  
elif region == "South_America":
    lat = [-60, -30]
    lon = [-70,-35] 
elif region == "Africa":
    lat = [-50, -20]
    lon = [5,45] 
elif region == "Quiescent":
    lat = [-42.5, -17.5]
    lon = [-155,-120] 
elif region == "Pacific":
    lat = [-35, 35]
    lon = [-230,-80]     
elif region == "Indian":
    lat = [-30, 28]
    lon = [30,120]  
    

    
s_train = lag*hist
e_train = s_train + N_samples*interval
e_test = e_train + interval*N_val



device = "cuda"


inpt_dict = {"1":["um","vm"],"3":["um","vm","Tm"],
            "9":["u","v"],"10":["u","v","T"]} 
extra_dict = {"1":["ur","vr"],"5":[],"9":["ur","vr","tau_u","tau_v"],
              "10":["tau_u","tau_v"],"11":["ur","vr","Tr"],
              "12":["tau_u","tau_v","t_ref"]} 
res_dict = {"1":["um","vm"],"2":["um","vm","Tm"],"3":["ur","vr"],
           "4":["ur","vr","Tr"],"5":["u","v"],"6":["u","v","T"]}


grids = xr.open_dataset('/scratch/zanna/data/CM2_grids/Grid_cm26_Vertices.nc')

grids = grids.sel({"yu_ocean":slice(lat[0],lat[1]),"xu_ocean":slice(lon[0],lon[1])})

area = torch.from_numpy(grids["area_C"].to_numpy()).to(device="cpu")

inputs = inpt_dict[exp_num_in]
extra_in = extra_dict[exp_num_extra]
res_in = res_dict[exp_num_res]

str_in = "".join([i + "_" for i in inputs])
str_ext = "".join([i + "_" for i in extra_in])
str_res = "".join([i + "_" for i in res_in])

print("inputs: " + str_in)
print("extra inputs: " + str_ext)
print("residuals: " + str_res)

N_extra = len(extra_in)
N_in = len(inputs)
N_res = len(res_in)
N_out = N_in

num_in = int((hist+1)*N_in + N_extra + N_res)

inputs, extra_in, outputs = gen_data(inputs,extra_in+res_in,inputs,lag,factor,region)

wet = xr.zeros_like(inputs[0][0])
# inputs[0][0,12,12] = np.nan
for data in inputs:
    wet +=np.isnan(data[0])
wet_nan = xr.where(wet!=0,np.nan,1).to_numpy()
wet = xr.where(wet==0,np.nan,0)    
wet = np.isnan(wet)
wet = np.nan_to_num(wet.to_numpy())
wet = torch.from_numpy(wet).type(torch.float32).to(device="cpu")
wet_bool = np.array(wet.cpu()).astype(bool)

time_vec = inputs[0].time.data

# %%
data_in_test = gen_data_in_test(0,e_test,N_test,lag,hist,inputs,extra_in)

data_out_test = gen_data_out_test(0,e_test,N_test,lag,hist,outputs)

test_data = data_CNN_Dynamic(data_in_test,data_out_test,wet.to(device = "cpu"),device="cuda") 

time_test = time_vec[e_test:(e_test+lag*N_test)]

mean_out = test_data.norm_vals['m_out']  
std_out = test_data.norm_vals['s_out']  

# %%
for step in range(4,8):
        
    N_s = "750"
    if step == 5:
        N_s = "500"
    steps = str(step)


    model_res = U_net([int(num_in-N_res),64,128,256,512],N_res,wet.to(device=device))


    model_res.to(device=device);
    model_res.load_state_dict(torch.load("/scratch/as15415/Emulation/Networks/U_net_Parallel_"+region+"_res_in_um_vm_Tm_ext_tau_u_tau_v_t_ref_N_train_"+N_s+"_region_"+region+"_N_samples_"+N_s+"_steps_"+steps+".pt",map_location=torch.device(device)))

    if network == "CNN":
        model = CNN(num_in = num_in, num_out = N_out, num_channels = 64,num_layers = 5,kernel_size=3)
    elif network == "U_net":
        model = U_net([num_in,64,128,256,512],N_out,wet.to(device=device))
    elif network == "U_net_RK":
        model = U_net_RK([num_in,64,128,256,512],N_out,wet)
    model.to(device=device);
    model.load_state_dict(torch.load("/scratch/as15415/Emulation/Networks/U_net_Parallel_"+region+"_mean_inum_vm_Tm_ext_tau_u_tau_v_t_ref_N_train_"+N_s+"_region_"+region+"_N_samples_"+N_s+"_steps_"+steps+".pt",map_location=torch.device(device)))


    model_pred, res_pred = recur_pred_joint_res(1000,test_data,model,model_res,hist,N_in,N_extra,N_res)

    da = xr.DataArray(
        data=model_pred,
        dims=["time","x", "y","var"],
    )

    da.to_zarr("/scratch/as15415/Emulation/Preds/Pred_Joint_"+region+"_in_"+str_in+"ext_"+str_ext+"N_samples_750_steps_"+str(step)+".zarr")
    
    da = xr.DataArray(
        data=res_pred,
        dims=["time","x", "y","var"],
    )

    da.to_zarr("/scratch/as15415/Emulation/Preds/Pred_Joint_Res_"+region+"_in_"+str_in+"ext_"+str_ext+"N_samples_750_steps_"+str(step)+".zarr")

# %%
1

# %%



