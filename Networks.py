import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import cartopy.crs as ccrs
import networkx as nx
import torch
import torch.nn as nn
import torch.utils.data as data
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from sklearn.neighbors import KernelDensity
from Utils import *

#Class defined to store information about the grid and corresponding graph of data. Importantly produces the adjacency matrices 
#and keeps track of what is land vs ocean 
            

    
class CNN(torch.nn.Module):

    def __init__(self,num_in = 2, num_out = 2,num_channels = 64, num_layers=6,kernel_size = 5):
        super().__init__()
        self.N_in = num_in

        layers = []
        layers.append(torch.nn.Conv2d(num_in,num_channels,kernel_size,padding='same'))
        layers.append(torch.nn.ReLU())
        for _ in range(num_layers):
            layers.append(torch.nn.Conv2d(num_channels,num_channels,kernel_size,padding='same'))
            layers.append(torch.nn.ReLU())              
        layers.append(torch.nn.Conv2d(num_channels,num_out,kernel_size,padding='same'))

        self.layers = nn.ModuleList(layers)
        #self.layers = nn.ModuleList(layer)

    def forward(self,fts):
        for l in self.layers:
            fts= l(fts)
        return fts

class CNN_linear(torch.nn.Module):

    def __init__(self,num_in = 2, num_out = 2,num_channels = 64, num_layers=6,kernel_size = 5):
        super().__init__()
        self.N_in = num_in

        layers = []
        layers.append(torch.nn.Conv2d(num_in,num_channels,kernel_size,padding='same'))
        for _ in range(num_layers):
            layers.append(torch.nn.Conv2d(num_channels,num_channels,kernel_size,padding='same'))
        layers.append(torch.nn.Conv2d(num_channels,num_out,kernel_size,padding='same'))

        self.layers = nn.ModuleList(layers)
        #self.layers = nn.ModuleList(layer)

    def forward(self,fts):
        for l in self.layers:
            fts= l(fts)
        return fts
    
class CNN_AB(torch.nn.Module):

    def __init__(self,num_in = 2, num_out = 2,num_channels = 64, num_layers=6,kernel_size = 5):
        super().__init__()
        self.N_in = int(num_in/2)
        self.num_out = num_out
        layers = []
        layers.append(torch.nn.Conv2d(self.N_in,num_channels,kernel_size,padding='same'))
        layers.append(torch.nn.ReLU())
        for _ in range(num_layers):
            layers.append(torch.nn.Conv2d(num_channels,num_channels,kernel_size,padding='same'))
            layers.append(torch.nn.ReLU())              
        layers.append(torch.nn.Conv2d(num_channels,num_out,kernel_size,padding='same'))

        self.layers = nn.ModuleList(layers)
        #self.layers = nn.ModuleList(layer)

    def forward(self,fts):
        u = fts[:,:self.num_out]
        fts_u = fts[:,:self.N_in]
        fts_old = fts[:,self.N_in:]
        
        for l in self.layers:
            fts_u = l(fts_u)
            fts_old = l(fts_old)
            
        return u + 3/2*fts_u - 1/2*fts_old

    
class CNN_Euler(torch.nn.Module):

    def __init__(self,num_in = 2, num_out = 2,num_channels = 64, num_layers=6,kernel_size = 5):
        super().__init__()
        self.N_in = num_in
        self.num_out = num_out
        layers = []
        layers.append(torch.nn.Conv2d(num_in,num_channels,kernel_size,padding='same'))
        layers.append(torch.nn.ReLU())
        for _ in range(num_layers):
            layers.append(torch.nn.Conv2d(num_channels,num_channels,kernel_size,padding='same'))
            layers.append(torch.nn.ReLU())              
        layers.append(torch.nn.Conv2d(num_channels,num_out,kernel_size,padding='same'))

        self.layers = nn.ModuleList(layers)
        #self.layers = nn.ModuleList(layer)

    def forward(self,fts):
        u = fts[:,:self.num_out]
        for l in self.layers:
            fts= l(fts)
            
        return u + fts
    
    
class CNN_RK(torch.nn.Module):

    def __init__(self,num_in = 2, num_out = 2,num_channels = 64, num_layers=6,kernel_size = 5):
        super().__init__()
        self.N_in = num_in
        self.num_out = num_out
        layers = []
        layers.append(torch.nn.Conv2d(num_in,num_channels,kernel_size,padding='same'))
        layers.append(torch.nn.ReLU())
        for _ in range(num_layers):
            layers.append(torch.nn.Conv2d(num_channels,num_channels,kernel_size,padding='same'))
            layers.append(torch.nn.ReLU())              
        layers.append(torch.nn.Conv2d(num_channels,num_out,kernel_size,padding='same'))

        self.layers = nn.ModuleList(layers)
        #self.layers = nn.ModuleList(layer)
    
    def step(self,fts):
        for l in self.layers:
            fts= l(fts) 
        return fts
    
    def forward(self,fts):
        u = fts[:,:self.num_out]
        
        f1 = self.step(fts)
        
        fts_2 = fts.clone()
        fts_2[:,:self.num_out] = fts_2[:,:self.num_out] + 1/2*f1
        f2 = self.step(fts_2)

        fts_3 = fts.clone()
        fts_3[:,:self.num_out] = fts_3[:,:self.num_out] + 1/2*f2
        f3 = self.step(fts_3)
        
        fts_4 = fts.clone()
        fts_4[:,:self.num_out] = fts_4[:,:self.num_out] + f3
        f4 = self.step(fts_4)
        
        return u + 1/6*(f1 + 2*f2 + 2*f3 +f4)
    
def mean_KE(data,area):
    square = .5*(data**2).sum(dim=1)
    KE = (area*square).sum(dim=[1,2])/area.sum()
    return KE

def loss_KE(data,out,area):
    square = .5*(data[:,:2]**2).sum(dim=1)
    KE_true = (area*square).sum(dim=[1,2])/area.sum()
    
    square = .5*(out[:,:2]**2).sum(dim=1)
    KE_out = (area*square).sum(dim=[1,2])/area.sum()  
    
    return ((KE_true - KE_out)**2).mean() 

def loss_T(data,out,area):
    square = .5*(data**2).sum(dim=1)
    KE_true = (area*square).sum(dim=[1,2])/area.sum()
    
    square = .5*(out**2).sum(dim=1)
    KE_out = (area*square).sum(dim=[1,2])/area.sum()  
    
    return ((KE_true - KE_out)**2).mean() 

def loss_KE_zonal(data,out,area):
    square = .5*(data[:,:2]**2).sum(dim=1)
    KE_true = (area*square).sum(dim=[2])/area.sum()
    
    square = .5*(out[:,:2]**2).sum(dim=1)
    KE_out = (area*square).sum(dim=[2])/area.sum()  
    
    return ((KE_true - KE_out)**2).mean() 

def loss_KE_pointwise(data,out):
    return ((data[:,:2]**2 - out[:,:2]**2)**2).mean() 
'''
def loss_KE_pointwise(data,out, N_level = 1):
    loss = ((data[:,0]**2+data[:,1]**2 - out[:,0]**2-out[:,1]**2)**2).mean() 
    for i in range(1,N_level):
        KE_true = out[:,int(3*i)]**2+out[:,int(3*i+1)]**2
        KE_pred = data[:,int(3*i)]**2+data[:,int(3*i+1)]**2
        loss +=((KE_true - KE_pred)**2).mean()
    return loss
'''
def loss_KE_pointwise_area(data,out,area):
    return ((((data[:,:2]**2).sum(dim=1) - (out[:,:2]**2).sum(dim=1))**2)*area).sum()/area.sum()

def loss_KE_pointwise_mae(data,out):
    return (torch.abs(data[:,:2]**2 - out[:,:2]**2)).mean() 

def lap_loss_diff(data,out,dx,Nb,wet_lap):
    lap_true = compute_laplacian(data[:,2],dx,Nb,wet_lap)
    lap_pred = compute_laplacian(out[:,2],dx,Nb,wet_lap)
    
    return torch.abs(torch.abs(lap_true).mean()-torch.abs(lap_pred).mean())

def lap_loss_reg(out,dx,Nb,wet_lap):
    lap_pred = compute_laplacian(out[:,2],dx,Nb,wet_lap)
    return torch.abs(lap_pred).mean()


def loss_spectral(pred,true,Kc):
    true_fft = torch.fft.rfft(true)
    pred_fft = torch.fft.rfft(pred)
    true_fft = true_fft.mean(dim = 2)
    pred_fft = pred_fft.mean(dim = 2)
    
    loss = (torch.linalg.vector_norm(true_fft[:,0,Kc:],dim=1)-
            torch.linalg.vector_norm(pred_fft[:,0,Kc:],dim=1))**2
    loss += (torch.linalg.vector_norm(true_fft[:,1,Kc:],dim=1)-
            torch.linalg.vector_norm(pred_fft[:,1,Kc:],dim=1))**2
    return loss.mean()
    

def train_CNN(model, train_loader, test_loader,N_in,N_extra,hist,num_epochs,loss_fn, optimizer,steps,weight):
    mse = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        for data in train_loader:

            optimizer.zero_grad()
            outs = model(data[0])
            outs = outs
            loss = loss_fn(data[1], outs)*weight[0]
            if len(weight) == 1:
                loss.backward()
            else:
                for step in range(1,steps):

                    if (step == 1) or (hist == 0):
                        step_in = torch.concat((outs,data[int(step*2)][:,N_in:]),1)
                        outs_old = outs
                    elif (step > 1) and (hist == 1):
                        step_in = torch.concat((outs,data[int(step*2)][:,N_in:(N_in+N_extra)],outs_old),1)
                        outs_old = outs
                    else:
                        step_in = torch.concat((outs,data[int(step*2)][:,N_in:(N_in+N_extra)],
                                                outs_old,step_in[:,(N_in+N_extra):-N_in]),1)
                        outs_old = outs

                    outs = model(step_in)
                    outs = outs

                    loss += loss_fn(data[int(step*2+1)], outs)*weight[step]
                loss.backward()

            optimizer.step()
        for data, label in test_loader:
            with torch.no_grad():
                outs = model(data)
                loss_val = mse(outs, label)

        print({"Train_loss": loss, "Val_loss": loss_val})
        
        
def train_CNN_Res(model,model_res, train_loader, test_loader,N_in,N_extra,N_res,hist,num_epochs,loss_fn, optimizer,optimizer_res,steps,weight,alpha):
    mse = torch.nn.MSELoss()
    
    slices = list(range(N_in))
    
    for i in range(N_in+N_res,N_in+N_res+N_extra):
        slices.append(i)
    
    for i in range(N_in+N_res+N_extra,N_in+N_res+N_extra + hist*N_in):
        slices.append(i)
        
    res_inds = []
    
    for i in range(N_in,N_in+N_res):
        res_inds.append(i)
    
    for epoch in range(num_epochs):
        for data in train_loader:
            inpt = data[0]
            optimizer.zero_grad()
            optimizer_res.zero_grad()
            
            res = model_res(data[0][:,slices,])
            inpt[:,res_inds] = res
            outs = model(inpt)
            
            loss_res = loss_fn(data[0][:,res_inds],res)*weight[0] 
            loss = loss_fn(data[1], outs)*weight[0] 
            if len(weight) == 1:
                loss = loss + loss_res*alpha
                loss.backward()
            else: 
                for step in range(1,steps):
                    if (step == 1) or (hist == 0):
                        step_in = torch.concat((outs,data[int(step*2)][:,N_in:]),1)
                        outs_old = outs
                    elif (step > 1) and (hist == 1):
                        step_in = torch.concat((outs,data[int(step*2)][:,N_in:(N_in+N_extra+N_res)],outs_old),1)
                        outs_old = outs
                    else:
                        step_in = torch.concat((outs,data[int(step*2)][:,N_in:(N_in+N_extra+N_res)],
                                                outs_old,step_in[:,(N_in+N_extra+N_res):-N_in]),1)
                        outs_old = outs                    
                    
                        
                    res = model_res(step_in[:,slices])
                    step_in[:,res_inds] = res
                    outs = model(step_in)
                    loss += loss_fn(data[int(step*2+1)], outs)*weight[step] 
                    loss_res += loss_fn(data[int(step*2)][:,res_inds],res)*weight[step] 
                                
#                 loss_res.backward()
                loss = loss + loss_res*alpha
                loss.backward()

            optimizer.step()
            optimizer_res.step()
            
        for data, label in test_loader:
            with torch.no_grad():
                res = model_res(data[:,slices,])
                inpt[:,res_inds] = res  
                outs = model(inpt)
                loss_val = mse(outs, label)
            
        print({"Train_loss": loss, "Val_loss": loss_val})


class Conv_block(torch.nn.Module):

    def __init__(self,num_in = 2, num_out = 2,kernel_size = 3, num_layers=2, pad = "constant"):
        super().__init__()
        self.N_in = num_in
        self.N_pad = int((kernel_size-1)/2)
        self.pad = pad
        
        layers = []
        layers.append(torch.nn.Conv2d(num_in,num_out,kernel_size))
        layers.append(torch.nn.BatchNorm2d(num_out))        
        layers.append(torch.nn.ReLU())
        for _ in range(num_layers-1):
            layers.append(torch.nn.Conv2d(num_out,num_out,kernel_size))
            layers.append(torch.nn.BatchNorm2d(num_out))
            layers.append(torch.nn.ReLU())              

        self.layers = nn.ModuleList(layers)
        #self.layers = nn.ModuleList(layer)

    def forward(self,fts):
        for l in self.layers:
            if isinstance(l,nn.Conv2d):
                fts = torch.nn.functional.pad(fts,(self.N_pad,self.N_pad,0,0),mode=self.pad)
                fts = torch.nn.functional.pad(fts,(0,0,self.N_pad,self.N_pad),mode="constant")
            fts= l(fts)
        return fts

class U_net(torch.nn.Module):
    def __init__(self,ch_width,n_out,wet,kernel_size = 3,pad = "constant"):
        super().__init__()
        self.N_in = ch_width[0]
        self.N_out = ch_width[-1]
        self.wet = wet
        self.N_pad = int((kernel_size-1)/2)
        self.pad = pad

        # going down
        layers = []
        for a,b in pairwise(ch_width):
            layers.append(Conv_block(a,b,pad=pad))
            layers.append(nn.MaxPool2d(2))
        layers.append(Conv_block(b,b,pad=pad))    
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        ch_width.reverse()
        for a,b in pairwise(ch_width[:-1]):
            layers.append(Conv_block(a,b,pad=pad))
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        layers.append(Conv_block(b,b,pad=pad))    
        layers.append(torch.nn.Conv2d(b,n_out,kernel_size))

        
        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width)-1)
        
        #self.layers = nn.ModuleList(layer)

    def forward(self,fts):
        temp = []
        for i in range(self.num_steps):
            temp.append(None)
        count = 0
        for l in self.layers:
            crop = fts.shape[2:]
            if isinstance(l,nn.Conv2d):
                fts = torch.nn.functional.pad(fts,(self.N_pad,self.N_pad,0,0),mode=self.pad)
                fts = torch.nn.functional.pad(fts,(0,0,self.N_pad,self.N_pad),mode="constant")
            fts= l(fts)
            if count < self.num_steps:
                if isinstance(l,Conv_block):
                    temp[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if isinstance(l,nn.Upsample):
                    crop = np.array(fts.shape[2:])
                    shape = np.array(temp[int(2*self.num_steps-count-1)].shape[2:])
                    pads = (shape - crop)
                    pads = [pads[1]//2, pads[1]-pads[1]//2,
                            pads[0]//2, pads[0]-pads[0]//2]
                    fts = nn.functional.pad(fts,pads)
                    fts += temp[int(2*self.num_steps-count-1)]
                    count += 1
        return torch.mul(fts,self.wet)  
    
    
class U_net_RK(torch.nn.Module):
    def __init__(self,ch_width,n_out,wet,set_size = True,kernel_size = 3,pad = "constant"):
        super().__init__()
        self.N_in = ch_width[0]
        self.N_out = ch_width[-1]
        self.num_out = n_out
        self.wet = wet
        self.N_pad = int((kernel_size-1)/2)
        self.pad = pad        
        # going down
        layers = []
        for a,b in pairwise(ch_width):
            layers.append(Conv_block(a,b,pad=pad))
            layers.append(nn.MaxPool2d(2))
        layers.append(Conv_block(b,b,pad=pad))    
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        ch_width.reverse()
        for a,b in pairwise(ch_width[:-1]):
            layers.append(Conv_block(a,b,pad=pad))
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        layers.append(Conv_block(b,b,pad=pad))    
        layers.append(torch.nn.Conv2d(b,n_out,kernel_size))

        
        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width)-1)
        #self.layers = nn.ModuleList(layer)

    def step(self,fts):
        temp = []
        for i in range(self.num_steps):
            temp.append(None)
        count = 0
        for l in self.layers:
            crop = fts.shape[2:]
            if isinstance(l,nn.Conv2d):
                fts = torch.nn.functional.pad(fts,(self.N_pad,self.N_pad,0,0),mode=self.pad)
                fts = torch.nn.functional.pad(fts,(0,0,self.N_pad,self.N_pad),mode="constant")
            fts= l(fts)
            if count < self.num_steps:
                if isinstance(l,Conv_block):
                    temp[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if isinstance(l,nn.Upsample):
                    crop = np.array(fts.shape[2:])
                    shape = np.array(temp[int(2*self.num_steps-count-1)].shape[2:])
                    pads = (shape - crop)
                    pads = [pads[1]//2, pads[1]-pads[1]//2,
                            pads[0]//2, pads[0]-pads[0]//2]
                    fts = nn.functional.pad(fts,pads)
                    fts += temp[int(2*self.num_steps-count-1)]
                    count += 1
        return fts
    
    def forward(self,fts):
        u = fts[:,:self.num_out]
        
        f1 = self.step(fts)
        
        fts_2 = fts.clone()
        fts_2[:,:self.num_out] = fts_2[:,:self.num_out] + 1/2*f1
        f2 = self.step(fts_2)

        fts_3 = fts.clone()
        fts_3[:,:self.num_out] = fts_3[:,:self.num_out] + 1/2*f2
        f3 = self.step(fts_3)
        
        fts_4 = fts.clone()
        fts_4[:,:self.num_out] = fts_4[:,:self.num_out] + f3
        f4 = self.step(fts_4)
        
        return torch.mul(u + 1/6*(f1 + 2*f2 + 2*f3 +f4),self.wet)
    
class U_net_PEC(torch.nn.Module):
    def __init__(self,ch_width,n_out,wet,set_size = True,kernel_size = 3,pad = "constant"):
        super().__init__()
        self.N_in = ch_width[0]
        self.N_out = ch_width[-1]
        self.num_out = n_out
        self.wet = wet
        self.N_pad = int((kernel_size-1)/2)
        self.pad = pad        
        # going down
        layers = []
        for a,b in pairwise(ch_width):
            layers.append(Conv_block(a,b,pad=pad))
            layers.append(nn.MaxPool2d(2))
        layers.append(Conv_block(b,b,pad=pad))    
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        ch_width.reverse()
        for a,b in pairwise(ch_width[:-1]):
            layers.append(Conv_block(a,b,pad=pad))
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        layers.append(Conv_block(b,b,pad=pad))    
        layers.append(torch.nn.Conv2d(b,n_out,kernel_size))

        
        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width)-1)
        #self.layers = nn.ModuleList(layer)

    def step(self,fts):
        temp = []
        for i in range(self.num_steps):
            temp.append(None)
        count = 0
        for l in self.layers:
            crop = fts.shape[2:]
            if isinstance(l,nn.Conv2d):
                fts = torch.nn.functional.pad(fts,(self.N_pad,self.N_pad,0,0),mode=self.pad)
                fts = torch.nn.functional.pad(fts,(0,0,self.N_pad,self.N_pad),mode="constant")
            fts= l(fts)
            if count < self.num_steps:
                if isinstance(l,Conv_block):
                    temp[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if isinstance(l,nn.Upsample):
                    crop = np.array(fts.shape[2:])
                    shape = np.array(temp[int(2*self.num_steps-count-1)].shape[2:])
                    pads = (shape - crop)
                    pads = [pads[1]//2, pads[1]-pads[1]//2,
                            pads[0]//2, pads[0]-pads[0]//2]
                    fts = nn.functional.pad(fts,pads)
                    fts += temp[int(2*self.num_steps-count-1)]
                    count += 1
        return fts
    
    def forward(self,fts):
        u = fts[:,:self.num_out]
        f1 = self.step(fts)
        fts_2 = fts.clone()
        fts_2[:,:self.num_out] = fts_2[:,:self.num_out] + 1/2*f1
        f2 = self.step(fts_2)
        return torch.mul(u + f2,self.wet)  
    
class U_net_3D(torch.nn.Module):
    def __init__(self,ch_width,n_out,wet,n_var,kernel_size = 3,pad = "constant"):
        super().__init__()
        self.N_in = ch_width[0]
        self.N_out = ch_width[-1]
        self.wet = wet
        self.N_pad = int((kernel_size-1)/2)
        self.pad = pad
        self.N_depth = len(wet)
        self.N_var = n_var

        # going down
        layers = []
        for a,b in pairwise(ch_width):
            layers.append(Conv_block(a,b,pad=pad))
            layers.append(nn.MaxPool2d(2))
        layers.append(Conv_block(b,b,pad=pad))    
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        ch_width.reverse()
        for a,b in pairwise(ch_width[:-1]):
            layers.append(Conv_block(a,b,pad=pad))
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        layers.append(Conv_block(b,b,pad=pad))    
        layers.append(torch.nn.Conv2d(b,n_out,kernel_size))

        
        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width)-1)
        
        #self.layers = nn.ModuleList(layer)

    def forward(self,fts):
        temp = []
        for i in range(self.num_steps):
            temp.append(None)
        count = 0
        for l in self.layers:
            crop = fts.shape[2:]
            if isinstance(l,nn.Conv2d):
                fts = torch.nn.functional.pad(fts,(self.N_pad,self.N_pad,0,0),mode=self.pad)
                fts = torch.nn.functional.pad(fts,(0,0,self.N_pad,self.N_pad),mode="constant")
            fts= l(fts)
            if count < self.num_steps:
                if isinstance(l,Conv_block):
                    temp[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if isinstance(l,nn.Upsample):
                    crop = np.array(fts.shape[2:])
                    shape = np.array(temp[int(2*self.num_steps-count-1)].shape[2:])
                    pads = (shape - crop)
                    pads = [pads[1]//2, pads[1]-pads[1]//2,
                            pads[0]//2, pads[0]-pads[0]//2]
                    fts = nn.functional.pad(fts,pads)
                    fts += temp[int(2*self.num_steps-count-1)]
                    count += 1
        for i in range(self.N_depth):
            fts[:,i*self.N_var:self.N_var*(i+1)] = torch.mul(fts[:,i*self.N_var:self.N_var*(i+1)],self.wet[i])
        return fts   
    
    
