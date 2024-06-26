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
import random

exp_num_in = sys.argv[1]
exp_num_extra = sys.argv[2]
exp_num_out = sys.argv[3]


mse = torch.nn.MSELoss()
args = {}

region = "Gulf_Stream"  
network = "U_net"

interval = 1

N_samples = 750
N_val = 100
N_test = 100

factor = 10

hist = 1

lag = 1

lam = 0

steps = 1


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

    
    
torch.manual_seed(rand_seed)
random.seed(rand_seed)
np.random.seed(rand_seed)

    
    
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



device = "cpu"


inpt_dict = {"1":["um","vm"],"2":["um","vm","ur","vr"],"3":["um","vm","Tm"],
            "4":["um1","vm1","Tm1"],"5":["um1","vm1","Tm1","um2","vm2","Tm2"]} 
extra_dict = {"1":["um","vm","Tm"]
              ,"5":[],"12":["tau_u","tau_v","t_ref"],
             "13":["ur","vr","Tr","tau_u","tau_v","t_ref"]} 
out_dict = {"1":["um","vm"],"2":["um","vm","Tm"],"3":["ur","vr"],
           "4":["ur","vr","Tr"],
           "5":["um1","vm1","Tm1"],"6":["um1","vm1","Tm1","um2","vm2","Tm2"]}

grids = xr.open_dataset('/scratch/zanna/data/CM2_grids/Grid_cm26_Vertices.nc')

if region == "global_25":
    grids = xr.open_dataset('/scratch/zanna/data/CM2_grids/Grid_cm25_Vertices.nc')

elif "global" in region:
    grids = coarse_grid(grids,factor)

else:
    grids = grids.sel({"yu_ocean":slice(lat[0],lat[1]),"xu_ocean":slice(lat[0],lon[1])})


area = torch.from_numpy(grids["area_C"].to_numpy()).to(device=device)

inputs = inpt_dict[exp_num_in]
extra_in = extra_dict[exp_num_extra]
outputs = out_dict[exp_num_out]

str_in = "".join([i + "_" for i in inputs])
str_ext = "".join([i + "_" for i in extra_in])
str_out = "".join([i + "_" for i in outputs])

print("inputs: " + str_in)
print("extra inputs: " + str_ext)
print("outputs: " + str_out)

N_extra = len(extra_in)
N_in = len(inputs)
N_out = len(outputs)

num_in = int((hist+1)*N_in + N_extra)

if "global" in region:
    if factor < 10 or factor >20:
        factor_str = "0" + str(factor)
    elif factor%10 != 0:
        factor_str = str(int(np.floor(factor/10))) + "_" +str(factor%10)
    elif factor%10 == 0:
        factor_str = str(int(np.floor(factor/10)))
    inputs, extra_in, outputs = gen_data_global(inputs,extra_in,outputs,lag,factor_str)
        
else:
    inputs, extra_in, outputs = gen_data_3d(inputs,extra_in,outputs,lag,factor,region)

wet = xr.zeros_like(inputs[0][0])
# inputs[0][0,12,12] = np.nan
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
args["num_in"] = num_in
args["str_in"] = str_in
args["str_ext"] = str_ext 
args["str_out"] = str_out


os.environ['MASTER_ADDR'] = 'localhost' 
os.environ['MASTER_PORT'] = '1324' 

args ["load"] = False

data_in_train = []
data_out_train = []

for i in range(steps):
    data_in_train.append(gen_data_in(i,s_train,e_train,interval,lag,hist,inputs,extra_in))
    data_out_train.append(gen_data_out(i,s_train,e_train,lag,interval,outputs))

train_data = data_CNN_steps_Dynamic(data_in_train,data_out_train,steps,wet,device=device)       

data_in_val = gen_data_in(0,e_train,e_test,interval,lag,hist,inputs,extra_in)  
data_out_val = gen_data_out(0,e_train,e_test,lag,interval,outputs)

val_data = data_CNN_Dynamic(data_in_val,data_out_val,wet,device=device)       

args["train_data"] = train_data
args["val_data"] = val_data
args["save_model"] = True
args["World_Size"] = 2
args["epochs"] = 15
args["batch_size"] = 6


lam = lam/1000
args["lam"] = lam
args["step_weights"] = [[1],[.2,.8],[.2,.2,.6],[0,0,.2,.8],[0,0,.1,.2,.7],[0,0,0,.1,.2,.7],[0,0,0,0,.1,.2,.7]]
args["step_lrs"] = [1e-4,5e-5,1e-5,1e-5,5e-6,5e-6,5e-6]


if __name__ == '__main__':
    torch.multiprocessing.spawn(worker_vary_steps, nprocs=args["World_Size"], args=(args,))   
