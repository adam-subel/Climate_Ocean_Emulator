# %%
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

import cartopy.crs as ccrs

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
from torch.utils.data import Dataset, DataLoader
import os 
import sys

os.environ['MKL_THREADING_LAYER'] = 'GNU'


# %%
exp_num_in = "3"
exp_num_extra = "12"
exp_num_res = "4"

args = {}

lateral = True

region = "Gulf_Stream_Ext"  
network = "U_net"

interval = 1

N_samples = 100
N_val = 100
N_test = 100

factor = 10

hist = 0

lag = 1

lam = 0

steps = 1

Nb = 4

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
elif region == "Africa_Ext":
    lat = [-55, -15]
    lon = [-5,55]     
    
    
s_train = lag*hist
e_train = s_train + N_samples*interval
e_test = e_train + interval*N_val



device = "cpu"


inpt_dict = {"1":["um","vm"],"3":["um","vm","Tm"],
            "9":["u","v"],"10":["u","v","T"]} 
extra_dict = {"1":["ur","vr"],"5":[],"9":["ur","vr","tau_u","tau_v"],
              "10":["tau_u","tau_v"],"11":["ur","vr","Tr"],
              "12":["tau_u","tau_v","t_ref"]} 
res_dict = {"1":["um","vm"],"2":["um","vm","Tm"],"3":["ur","vr"],
           "4":["ur","vr","Tr"],"5":["u","v"],"6":["u","v","T"]}

grids = xr.open_dataset('/scratch/as15415/Data/CM2x_grids/Grid_cm25_Vertices.nc')

grids = grids.sel({"yu_ocean":slice(lat[0],lat[1]),"xu_ocean":slice(lon[0],lon[1])})

area = torch.from_numpy(grids["area_C"].to_numpy()).to(device=device)


inputs = inpt_dict[exp_num_in]
extra_in = extra_dict[exp_num_extra]
res_in = res_dict[exp_num_res]

str_in = "".join([i + "_" for i in inputs])
str_ext = "".join([i + "_" for i in extra_in])
str_res = "".join([i + "_" for i in res_in])

print("inputs: " + str_in)
print("extra inputs: " + str_ext)
print("residuals: " + str_res)



N_in = len(inputs)
N_atm = len(extra_in) 
N_res = len(res_in)
if lateral:
    N_extra = N_atm + N_res + N_in
else:
    N_extra = N_atm + N_res

N_out = N_in

num_in = int((hist+1)*N_in + N_extra)



inputs, extra_in, outputs = gen_data_025_lateral(inputs,extra_in+res_in,inputs,lag,lat,lon,Nb)


wet = xr.zeros_like(inputs[0][0])
for data in inputs:
    wet +=np.isnan(data[0])
wet = np.isnan(xr.where(wet==0,np.nan,0))
wet = np.nan_to_num(wet.to_numpy())
wet = torch.from_numpy(wet).type(torch.float32).to(device="cpu")

args["s_train"] = s_train
args["e_train"] = e_train
args["e_test"] = e_test
args["inputs"] = inputs
args["extra_in"] = extra_in
args["outputs"] = outputs
args["wet"] = wet
args["area"] = area
args["N_extra"] = N_extra
args["N_in"] = N_in
args["N_out"] = N_out
args["N_res"] = N_res
args["N_atm"] = N_atm
args["num_in"] = num_in
args["str_in"] = str_in
args["str_ext"] = str_ext
args["str_res"] = str_res
args["init_weights"] = False
args["Nb"] = Nb



args ["load"] = False

args["lateral"] = lateral
os.environ['MASTER_ADDR'] = 'localhost' 
os.environ['MASTER_PORT'] = str(np.random.randint(1400,1600)) 


# data_in_train = []
# data_out_train = []

# for i in range(steps):
#     data_in_train.append(gen_data_in(i,s_train,e_train,interval,lag,hist,inputs,extra_in))
#     data_out_train.append(gen_data_out(i,s_train,e_train,lag,interval,outputs))

# train_data = data_CNN_steps_Lateral(data_in_train,data_out_train,steps,wet,N_atm,Nb,device=device)       

data_in_val = gen_data_in(0,e_train,e_test,interval,lag,hist,inputs,extra_in)  
data_out_val = gen_data_out(0,e_train,e_test,lag,interval,outputs)

val_data = data_CNN_Lateral(data_in_val,data_out_val,wet,N_atm,Nb,device=device)       

args["val_data"] = val_data
args["save_model"] = True
args["World_Size"] = int(torch.cuda.device_count())
args["epochs"] = 10
args["batch_size"] = 5
args["loss_type"] = "KE"


lam = lam/1000
args["lam"] = lam
args["step_weights"] = [[1],[.2,.8],[.1,.2,.7],[.1,.1,.2,.7],[0,0,.1,.2,.7],[0,0,0,.1,.2,.7],[0,0,0,0,.1,.2,.7]]
args["step_lrs"] = [1e-4,5e-5,1e-5,1e-5,5e-6,5e-6,5e-6]


if __name__ == '__main__':
    torch.multiprocessing.spawn(worker_joint_vary_steps, nprocs=args["World_Size"], args=(args,))   


