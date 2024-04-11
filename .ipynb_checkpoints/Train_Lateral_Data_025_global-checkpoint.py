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


exp_num_in = 6
exp_num_extra = sys.argv[2]
exp_num_out = 10



mse = torch.nn.MSELoss()
args = {}

region = "global_21"  
network = "U_net_Global"

interval = 1

N_samples = 4000
N_val = 300
N_test = 500

factor = 10

hist = 0

lag = 1

lam = 50

steps = 4


Nb = 4

rand_seed = 1

lateral = False




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

str_video += "_global"
    
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
              "11":["t_ref"],"12":["tau_u","tau_v","t_ref"],
             "13":["ur","vr","Tr","tau_u","tau_v","t_ref"]} 
out_dict = {"1":["um","vm"],"2":["um","vm","Tm"],"3":["ur","vr"],
           "4":["ur","vr","Tr"],"5":["u","v"],"6":["u","v","T"]}

grids = xr.open_dataset('/scratch/zanna/data/CM2_grids/Grid_cm25_Vertices.nc')

if region == "global_25":
    grids = xr.open_dataset('/scratch/zanna/data/CM2_grids/Grid_cm25_Vertices.nc')

elif "global" in region:
    grids = coarse_grid(grids,factor)

else:
    grids = grids.sel({"yu_ocean":slice(lat[0],lat[1]),"xu_ocean":slice(lon[0],lon[1])})


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

N_atm = len(extra_in)
N_in = len(inputs)
N_extra = N_atm + N_in
N_out = len(outputs)

num_in = int((hist+1)*N_in + N_extra)


inputs, extra_in, outputs = gen_data_global(inputs,extra_in,outputs,lag)

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
args["N_atm"] = N_atm
args["num_in"] = num_in
args["str_in"] = str_in
args["str_ext"] = str_ext 
args["str_out"] = str_out
args["Nb"] = Nb

args["lateral"] = lateral
os.environ['MASTER_ADDR'] = 'localhost' 
os.environ['MASTER_PORT'] = str(np.random.randint(1000,1200)) 


# data_in_train = []
# data_out_train = []

# for i in range(steps):
#     data_in_train.append(gen_data_in(i,s_train,e_train,interval,lag,hist,inputs,extra_in))
#     data_out_train.append(gen_data_out(i,s_train,e_train,lag,interval,outputs))

# train_data = data_CNN_steps_Lateral(data_in_train,data_out_train,steps,wet,N_atm,Nb,device=device)       

data_in_val = gen_data_in(0,e_train,e_test,interval,lag,hist,inputs,extra_in)  
data_out_val = gen_data_out(0,e_train,e_test,lag,interval,outputs)

if args["lateral"]:
    val_data = data_CNN_Lateral(data_in_val,data_out_val,wet,N_atm,Nb,device=device,norms = norm_vals) 
else:
    val_data = data_CNN_Dynamic(data_in_val,data_out_val,wet,device=device,norms = norm_vals)   


args ["load"] = False

# args["train_data"] = train_data
args["val_data"] = val_data
args["save_model"] = True
args["World_Size"] = int(torch.cuda.device_count())
args["epochs"] = 8
args["batch_size"] = 4



lam = lam/1000
args["lam"] = lam
args["step_weights"] = [[1],[.2,.8],[.2,.2,.6],[.1,.1,.2,.8],[.1,.1,.1,.2,.7],[.1,.1,.1,.1,.2,.7],
                        [.1,.1,.1,.1,.1,.2,.7],[.1,.1,.1,.1,.1,.1,.2,.7],[.05,.05,.05,.05,.05,.05,.1,.2,.7],
                        [.05,.05,.05,.05,.05,.05,.05,.1,.2,.7],[.05,.05,.05,.05,.05,.05,.05,.05,.1,.2,.7]
                       ,[.05,.05,.05,.05,.05,.05,.05,.05,.05,.1,.2,.7]]

if steps > 12:
    for i in range(steps-12):
        args["step_weights"].append([.05]+args["step_weights"][-1])

args["step_lrs"] = [1e-5,1e-5,1e-5,1e-5,5e-6,5e-6,5e-6,5e-6,5e-6,5e-6,5e-6,5e-6,5e-6,5e-6,5e-6,5e-6,
                    5e-6,5e-6,5e-6,5e-6,5e-6,5e-6,5e-6]


if __name__ == '__main__':
    torch.multiprocessing.spawn(worker_vary_steps_data_fast, nprocs=args["World_Size"], args=(args,))   



