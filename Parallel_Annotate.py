import argparse
import torch
import torch.nn as nn
import torch_geometric
import numpy as np
import os
import time as time
from Data_Functions import *
from Networks import *
import time as time
from Utils import *
from torch.utils.data import Dataset, DataLoader


import torch.distributed as dist

#Additional forms of the data functions that do not include lateral boundaries

class data_CNN_Dynamic(torch.utils.data.Dataset):

    def __init__(self,data_in,data_out,wet,device = "cuda"):
        super().__init__()
        self.device = device        
        num_inputs = data_in.shape[3]
        num_outputs = data_out.shape[3]
        self.size = data_in.shape[0]
        
        data_in = np.nan_to_num(data_in)
        data_out = np.nan_to_num(data_out)
        
        std_data = np.nanstd(data_in,axis=(0,1,2))
        mean_data = np.nanmean(data_in,axis=(0,1,2)) 
        std_label = np.nanstd(data_out,axis=(0,1,2))
        mean_label = np.nanmean(data_out,axis=(0,1,2))
        
        self.wet = wet
        
        for i in range(num_inputs):
            data_in[:,:,:,i] = (data_in[:,:,:,i] - mean_data[i])/std_data[i]
        
        for i in range(num_outputs):
            data_out[:,:,:,i] = (data_out[:,:,:,i] - mean_label[i])/std_label[i]
            
        data_in = torch.from_numpy(data_in).type(torch.float32).to(device="cpu")
        data_out = torch.from_numpy(data_out).type(torch.float32).to(device="cpu")        
        

        std_dict = {'s_in':std_data,'s_out':std_label,'m_in':mean_data, 'm_out':mean_label}
        
        if wet == None:
            self.input = torch.swapaxes(torch.swapaxes(data_in,1,3),2,3)
            self.output = torch.swapaxes(torch.swapaxes(data_out,1,3),2,3)           
            
        else:
            self.input = torch.mul(torch.swapaxes(torch.swapaxes(data_in,1,3),2,3),wet)
            self.output = torch.mul(torch.swapaxes(torch.swapaxes(data_out,1,3),2,3),wet)
        
        self.norm_vals = std_dict
    
    def set_device(self,device):
        self.device = device
    
    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_in = self.input[idx]
        label = self.output[idx]
        return data_in.to(device = self.device), label.to(device = self.device)

    
class data_CNN_steps_Dynamic(torch.utils.data.Dataset):

    def __init__(self,data_in,data_out,steps,wet = None,device = "cuda"):
        super().__init__()
        self.device = device
        steps = len(data_out)
        self.steps = steps
        num_inputs = data_in[0].shape[3]
        num_outputs = data_out[0].shape[3]
        self.size = data_in[0].shape[0]
        self.wet = wet
        
        for i in range(steps):
            data_out[i] = np.nan_to_num(data_out[i])
            data_in[i] = np.nan_to_num(data_in[i])
       
        std_data = np.nanstd(data_in[0],axis=(0,1,2))
        mean_data = np.nanmean(data_in[0],axis=(0,1,2)) 
        std_label = np.nanstd(data_out[0],axis=(0,1,2))
        mean_label = np.nanmean(data_out[0],axis=(0,1,2))
        

        for j in range(steps):
            for i in range(num_outputs):
                data_out[j][:,:,:,i] = (data_out[j][:,:,:,i] - mean_label[i])/std_label[i]
            for i in range(num_inputs):
                data_in[j][:,:,:,i] = (data_in[j][:,:,:,i] - mean_data[i])/std_data[i] 
                
        for j in range(steps):
            data_out[j] = torch.from_numpy(data_out[j]).type(torch.float32).to(device="cpu")    
            data_in[j] = torch.from_numpy(data_in[j]).type(torch.float32).to(device="cpu") 
       

        std_dict = {'s_in':std_data,'s_out':std_label,'m_in':mean_data, 'm_out':mean_label}
        
        if wet == None:
            for j in range(steps):
                data_out[j] = torch.swapaxes(torch.swapaxes(data_out[j],1,3),2,3)
                data_in[j] = torch.swapaxes(torch.swapaxes(data_in[j],1,3),2,3)
        else:
            for j in range(steps):
                data_out[j] = torch.mul(torch.swapaxes(torch.swapaxes(data_out[j],1,3),2,3),wet)
                data_in[j] = torch.mul(torch.swapaxes(torch.swapaxes(data_in[j],1,3),2,3),wet)
        
        self.input = data_in

        self.output = data_out
        self.norm_vals = std_dict
        
    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data = [self.input[0][idx].to(device = self.device),self.output[0][idx].to(device = self.device)]
        for k in range(1,self.steps):
            data.append(self.input[k][idx].to(device = self.device))
            data.append(self.output[k][idx].to(device = self.device))
        
        return tuple(data)

# Training functions over multiple time steps. Here we move the data onto the GPU as called
    

def train_parallel_Dynamic(model, train_loader,N_in,N_extra,hist,loss_fn, optimizer,steps,weight,device):
    mse = torch.nn.MSELoss()
    for data in train_loader:

        optimizer.zero_grad()
        outs = model(data[0].to(device = device))
        outs = outs
        loss = loss_fn(data[1].to(device = device), outs)*weight[0]
        if len(weight) == 1:
            loss.backward()
        else:
            for step in range(1,steps):

                if (step == 1) or (hist == 0):
                    step_in = torch.concat((outs,data[int(step*2)][:,N_in:].to(device = device)),1)
                    outs_old = outs
                elif (step > 1) and (hist == 1):
                    step_in = torch.concat((outs,data[int(step*2)][:,N_in:(N_in+N_extra)].to(device = device),outs_old),1)
                    outs_old = outs
                else:
                    step_in = torch.concat((outs,data[int(step*2)][:,N_in:(N_in+N_extra)].to(device = device),
                                            outs_old,step_in[:,(N_in+N_extra):-N_in]),1)
                    outs_old = outs

                outs = model(step_in)
                outs = outs

                loss += loss_fn(data[int(step*2+1)].to(device = device), outs)*weight[step]
            loss.backward()
    
        optimizer.step()
        torch.cuda.empty_cache()
    
        
        
def test_parallel_Dynamic(model, test_loader,device):
    mse = torch.nn.MSELoss()
    for data, label in test_loader:
            with torch.no_grad():
                outs = model(data.to(device = device))
                loss_val = mse(outs, label.to(device = device))
    return loss_val


# Worker function

        
def worker_vary_steps_data_fast(local_rank,args):
    # Initialize worker

    
    global_rank = local_rank*1
    dist.init_process_group(backend='nccl',world_size=args["World_Size"], rank=global_rank)
    
    device = torch.device("cuda:" + str(local_rank) if torch.cuda.is_available() else "cpu")
    device_name = "cuda:" + str(local_rank)
    
    args["area"] = args["area"].to(device)
    args["area"] = args["area"]/args["area"].min()
            
#   Same as before, but loads a subset of the data onto each thread
         
    data_in_train = []
    data_out_train = []
    for i in range(args["steps"]):
        offset = global_rank*args["interval"]
        data_in_train.append(gen_data_in(i,args["s_train"]+offset,args["e_train"],
                                         args["interval"]*args["World_Size"],args["lag"],
                                         args["hist"],args["inputs"],args["extra_in"]))
        data_out_train.append(gen_data_out(i,args["s_train"]+offset,args["e_train"],
                                           args["lag"],args["interval"]*args["World_Size"],
                                           args["outputs"]))
    
    if args["lateral"]:
        train_data = data_CNN_steps_Lateral(data_in_train,data_out_train,
                                            args["steps"],args["wet"],args["N_atm"],args["Nb"],device=device)  
    else:
        train_data = data_CNN_steps_Dynamic(data_in_train,data_out_train,
                                            args["steps"],args["wet"],device=device)      
    val_data = args["val_data"]

    
    dist.barrier()
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_data,
    num_replicas=1,
    rank=0
    )
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(
    val_data,
    num_replicas=args["World_Size"],
    rank=global_rank
    )
    
    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=args["batch_size"], sampler=train_sampler)
    test_loader = torch_geometric.loader.DataLoader(val_data, batch_size=args["batch_size"], sampler=test_sampler)

    if args["network"] == "CNN":
        model = CNN(num_in = args["num_in"], num_out = args["N_in"], num_channels = 64,num_layers = 5,kernel_size=3)
    elif args["network"] == "U_net":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(U_net([args["num_in"],64,128,256,512],
                                                                    args["N_in"],args["wet"].to(device)).to(device))

#   Define loss function
        
        
    lam = args["lam"]
    mse = torch.nn.MSELoss()
    loss = lambda out,pred:  mse(out,pred)*(1-lam) + loss_KE_pointwise(out,pred)*lam
    
    
#   Training the model 
    
    for k in range(args["steps"]):
        step_weights = args["step_weights"][k]
        lr = args["step_lrs"][k]
        optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-4, lr=lr)

        for epoch in range(args["epochs"]):
            train_sampler.set_epoch(int(epoch+k*args["epochs"]))
            test_sampler.set_epoch(int(epoch+k*args["epochs"]))
            
            train_parallel_Dynamic(model, train_loader,args["N_in"],args["N_extra"],
                           args["hist"],loss, optimizer,k,step_weights,device)

            v_loss = test_parallel_Dynamic(model,test_loader,device)

            torch.distributed.all_reduce(v_loss/args["World_Size"],op= dist.ReduceOp.SUM)

            if global_rank ==0:
                print("Epoch = {:2d}, Validation Loss = {:5.3f}".format(epoch+1,v_loss),flush=True)

    
    if global_rank ==0 and args["save_model"]:
        torch.save(model.state_dict(),'/scratch/as15415/Emulation/Networks/' + 'U_net_Parallel_Fast_'+args["region"]+'_Test_in_' + args["str_in"] + 'ext_' + args["str_ext"] +'_out'+args['str_out']+'N_train_' + str(args["N_samples"]) + args["str_video"] + '.pt')
        