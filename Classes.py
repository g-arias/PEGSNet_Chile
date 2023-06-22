#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:17:46 2020

@author: licciar
"""


import torch
import h5py
import numpy as np
import math
from torch import nn
from torch.utils.data import Sampler
import torch.distributed as dist
import random

class DistributedEvalSampler(Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)             # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch


class PEGSNet(nn.Module):
    def __init__(self, n_chan, n_stat, t_max, h_dim=8):
        super(PEGSNet, self).__init__()
        
        
        self.n_chan = n_chan
        self.height = t_max
        self.width =  n_stat
        self.h_dim  = h_dim
        
        drop_frac = 0.04
        self.cnn = nn.Sequential(nn.Conv2d(self.n_chan, 32, kernel_size=3,stride=1,padding=1),
                #nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Dropout2d(p = drop_frac),
                                                                       
                                    
                                    nn.Conv2d(32, 32, kernel_size=3,stride=1,padding=1),
                                    #nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Dropout2d(p = drop_frac),
                                    
                                    nn.Conv2d(32, 32, kernel_size=3,stride=1,padding=1),
                                    #nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Dropout2d(p = drop_frac),
                                                               
                                    
                                    nn.Conv2d(32, 32, kernel_size=3,stride=1,padding=1),
                                    #nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Dropout2d(p = drop_frac),
                                    nn.AdaptiveMaxPool2d((int(self.width/2),int(self.height/2)) ),

                                    
                                    # nn.Conv2d(32, 32, kernel_size=3,stride=1,padding=1),
                                    # #nn.BatchNorm2d(32),
                                    # nn.ReLU(),
                                    # nn.Dropout2d(p = drop_frac),
                                    # nn.AdaptiveMaxPool2d((int(self.width/4),int(self.height/4) ) ),



                                    nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=1),
                                    #nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Dropout2d(p = drop_frac),
                                    nn.AdaptiveMaxPool2d((int(self.width/4),int(self.height/4) ) ),
                                    
                                    nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1),
                                    #nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Dropout2d(p = drop_frac),
                                    nn.AdaptiveMaxPool2d((int(self.width/8),int(self.height/8) )),
                                    
         
                                    nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=1),
                                    #nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Dropout2d(p = drop_frac),
                                    nn.AdaptiveMaxPool2d((int(self.width/16),int(self.height/16) ) )
                                    
                                    
                                    )
        indim = int(int(self.height/16) * int(self.width/16) * 128)
        # print(indim)
        self.fcc = nn.Sequential( nn.Linear(indim, 512),
                #nn.BatchNorm1d(512),
                                  nn.ReLU(),
                                  nn.Dropout(p=drop_frac),
                                  
                                  nn.Linear(512,256),
                                  #nn.BatchNorm1d(256),
                                  nn.ReLU(),
                                  nn.Dropout(p=drop_frac),
                                  
                                  nn.Linear(256,3),
                                  nn.Tanh()
                       
            )
        



    def forward(self, x):
        x = self.cnn(x.double())
        x = x.view(x.size(0),-1)
        x = self.fcc(x)
        
        return x




class HDF5Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, n_stat, t_max, n_comp, database_path="../DATABASES/run_db13.hdf5"):
        'Initialization'
        super(HDF5Dataset, self).__init__()

        self.n_stat = n_stat
        self.t_max = t_max
        self.n_comp = n_comp
        self.database_path = database_path
        self.list_IDs = list_IDs
        #self.file = h5py.File(self.database_path, 'r')
  def open_hdf5(self):
        self.file = h5py.File(self.database_path, 'r')
        
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):

        if not hasattr(self, self.database_path):
            self.open_hdf5()        

        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        
        
        # Get a time shift at random between 0 and tmax
        shift = np.random.randint(0,self.t_max+1)
        
        # starting sample
        t1 = 350-shift
        # ending sample
        t2 = t1+self.t_max
        # samples with respect to origin
        t3 = self.t_max - shift
        
        # Get the relevant window
        X = np.array(self.file['pegs_w_noise_clip'][ID,:self.n_stat,t1:t2,:self.n_comp ])
        # get pwave arrivals
        pwav = np.array(self.file['ptime'][ID,:self.n_stat ])
        # Get STF value at this time
        STF_val = self.file["STF"][ID,t3]
        # Get the other parameters
        eq_params= np.array(self.file['eq_params'][ID,: ])
        
        
########################################################################################        
        # MUTE TRACES FOR WHICH we don't have P-WAVE arrivals
        #pdelay = 0.0
        #idmut = np.where( pwav - pdelay > t3)[0]
        #X[idmut,:,:] = 0.0
############################### 
        X[4,:,:] = 0.0
        # scale = 2.7e-8
        scale = 1.0e-8
        X = np.clip(X,-1.0*scale, scale)
        X /= scale
        # Got to swap axes because pytorch has channel first
        X = np.swapaxes(X,-1,0)
        
        
        # Minimum and maximum Mw in training database
        minmw= 5.5
        maxmw = 10.0
        
        # Very rarely the STF has nans. Mute this example
        if np.isnan(STF_val):
            X[:,:,:] = 0.0
            STF_val = 0.0
            #print("NAN")

        # Convert to pytorch tensor
        x = torch.from_numpy(X)
        
        
        # Convert STF at this time to Mw
        if STF_val>0:
            # GET current m0
            m0 = math.pow(10,1.5*eq_params[0]+9.1) * STF_val
            # Get corresponding Mw
            y_Mw = (np.log10(m0) - 9.1)/1.5
        else:
            y_Mw = minmw

        if y_Mw<minmw:
            y_Mw = minmw

        # Get labels for latitude and longitude
        y_lat = eq_params[2]
        y_lon = eq_params[1]
        
    
        # Minimum and maximum values for lables in training database
        minlat = -43.0
        maxlat = -22.0
        minlon = -74.6
        maxlon = -70.3
       
        # Normalize targets to [0,1] 
        #y_lat =  ((y_lat - minlat)/(maxlat-minlat)) 
        #y_lon =  ((y_lon - minlon)/(maxlon - minlon)) 
        #y_Mw =  ((y_Mw - minmw)/(maxmw - minmw)) 

        # Normalize targets to [-1,1]
        y_lat = 2* ((y_lat - minlat)/(maxlat-minlat)) -1.0
        y_lon = 2* ((y_lon - minlon)/(maxlon - minlon)) -1.0
        y_Mw = 2* ((y_Mw - minmw)/(maxmw - minmw)) -1.0

        # Normalize targets by just dividing by their maximum 
        #y_lat = y_lat/maxlat
        #y_lon = y_lon/maxlon
        #y_Mw = y_Mw/maxmw

        
        y = torch.from_numpy(np.array([y_Mw,y_lat,y_lon]))
        
        return x, y, index
    
    
        
    
