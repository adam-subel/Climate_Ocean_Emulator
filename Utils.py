import xarray as xr
import numpy as np
import xrft
from itertools import *


def compute_power_spectrum_iso(dx,dy,u):
    start = [i%2 for i in u.shape] 
    
    u = u[:,start[1]:,start[2]:].fillna(0)
   
    dx = dx[start[1]:,start[2]:]
    dx = dx.mean()
    dy = dy[start[1]:,start[2]:]
    dy = dy.mean()
    
    Nx = u.shape[2]
    Ny = u.shape[1]


    xline= (np.linspace(0,Nx*dx,Nx)) #in m
    yline= (np.linspace(0,Ny*dy,Ny))
    
    uline = xr.DataArray(u.chunk({"time":20,"yu_ocean":Ny,"xu_ocean":Nx}), coords=[u.time,yline,xline],dims=["time","yline","xline"])

    uiso2 = xrft.isotropic_power_spectrum(uline, dim=('xline','yline'), window='hann', nfactor=2, 
        truncate=True, detrend='linear', window_correction=True)

    uiso2['freq_r'] = uiso2.freq_r*1000*2*np.pi


    return uiso2





def KE_spectrum(dx,dy,u,v):
    u = xr.DataArray(
        data=np.expand_dims(u,0),
        dims=["time","yu_ocean", "xu_ocean"],
        coords=dict(
            time=np.array([1,]),        
            yu_ocean=(["yu_ocean"], dx.yu_ocean.data),
            xu_ocean=(["xu_ocean"], dx.xu_ocean.data)

        )
    )
    v = xr.DataArray(
        data=np.expand_dims(v,0),
        dims=["time","yu_ocean", "xu_ocean"],
        coords=dict(
            time=np.array([1,]),        
            yu_ocean=(["yu_ocean"], dx.yu_ocean.data),
            xu_ocean=(["xu_ocean"], dx.xu_ocean.data)

        )
    )    
    spec_u = compute_power_spectrum_iso(dx,dy,u)
    spec_v = compute_power_spectrum_iso(dx,dy,v)
    KE = (spec_u.mean("time")+spec_v.mean("time"))/2
    
    return KE


def KE_spectrum_long(dx,dy,u,v):
    u = xr.DataArray(
        data=u,
        dims=["time","yu_ocean", "xu_ocean"],
        coords=dict(
            time=np.arange(u.shape[0]),        
            yu_ocean=(["yu_ocean"], dx.yu_ocean.data),
            xu_ocean=(["xu_ocean"], dx.xu_ocean.data)

        )
    )
    v = xr.DataArray(
        data=v,
        dims=["time","yu_ocean", "xu_ocean"],
        coords=dict(
            time=np.arange(u.shape[0]),        
            yu_ocean=(["yu_ocean"], dx.yu_ocean.data),
            xu_ocean=(["xu_ocean"], dx.xu_ocean.data)

        )
    )    
    spec_u = compute_power_spectrum_iso(dx,dy,u)
    spec_v = compute_power_spectrum_iso(dx,dy,v)
    KE = (spec_u.mean("time")+spec_v.mean("time"))/2
    
    return KE

def compute_fft_2D(dx,dy,u):
    start = [i%2 for i in u.shape] 
    
    u = u[:,start[1]:,start[2]:].fillna(0)
   
    dx = dx[start[1]:,start[2]:]
    dx = dx.mean()
    dy = dy[start[1]:,start[2]:]
    dy = dy.mean()
    
    Nx = u.shape[2]
    Ny = u.shape[1]


    xline= (np.linspace(0,Nx*dx,Nx)) #in m
    yline= (np.linspace(0,Ny*dy,Ny))
    
    uline = xr.DataArray(u.chunk({"time":20,"yu_ocean":Ny,"xu_ocean":Nx}), coords=[u.time,yline,xline],dims=["time","yline","xline"])

    uiso2 = xrft.fft(uline, dim=('xline','yline'))

#     uiso2['freq_r'] = uiso2.freq_r*1000*2*np.pi


    return uiso2

def fft2d(dx,dy,u):
    u = xr.DataArray(
        data=u,
        dims=["time","yu_ocean", "xu_ocean"],
        coords=dict(
            time=np.arange(u.shape[0]),        
            yu_ocean=(["yu_ocean"], dx.yu_ocean.data),
            xu_ocean=(["xu_ocean"], dx.xu_ocean.data)

        )
    )

    spec_u = compute_fft_2D(dx,dy,u)
    
    return spec_u

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def compute_laplacian(T,dx,Nb,wet_lap):
#     h = 1/dx[Nb:-Nb,Nb:-Nb]**2
    lap = T[:,Nb:-Nb,Nb-1:-(Nb+1)] + T[:,Nb-1:-(Nb+1),Nb:-Nb] + \
    T[:,Nb:-Nb,Nb+1:-(Nb-1)] + T[:,Nb+1:-(Nb-1),Nb:-Nb] -\
    4*T[:,Nb:-Nb,Nb:-Nb]
    return lap*wet_lap

def compute_laplacian_wet(T,Nb):
    lap = T[Nb:-Nb,Nb-1:-(Nb+1)] + T[Nb-1:-(Nb+1),Nb:-Nb] + \
    T[Nb:-Nb,Nb+1:-(Nb-1)] + T[Nb+1:-(Nb-1),Nb:-Nb] -\
    4*T[Nb:-Nb,Nb:-Nb]
    return lap