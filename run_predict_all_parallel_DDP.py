#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:00:12 2021

@author: licciar
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:09:55 2021

@author: licciar
"""

import numpy as np
import matplotlib.pyplot as plt

import h5py
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, argparse
import torch
from Classes_predict import PEGSNet, HDF5Dataset, DistributedEvalSampler

import subprocess
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import idr_torch

# Use the same seed for testing
#torch.manual_seed(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#np.random.seed(0)


def setup():
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", init_method='env://',rank=idr_torch.rank, world_size=idr_torch.size)

def cleanup():
    dist.destroy_process_group()


"""parsing and configuration"""
def parse_args():
    desc = "Predict from real data"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model_name', type=str, default='./outputs/model_STF_duration_no_zeros_smoothL1/model_STF_duration_no_zeros_smoothL1_347k102k',
                        help='Path of the model to be tested (without extension)')
    
    
    parser.add_argument('--data_path', type=str, default='../DATABASES/run_db20_STF_duration.hdf5',
                        help='Path of the database to use')
    
    


    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    #   --PEGS_db_path       
    if not os.path.exists(args.model_name+'.pth'):
        print(args.model_name + '.pth does not exist')
        exit()
        
    if not os.path.exists(args.data_path):
        print(args.data_path + ' does not exist')
        exit()
        


    return args


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def main(args):
    setup()
    db_name = args.data_path #'../DATABASES/run_db20_STF_duration.hdf5'
    
    
    modelname= args.model_name #'./outputs/model_STF_duration_no_zeros_smoothL1/model_STF_duration_no_zeros_smoothL1_347k102k'
    test_idx = np.loadtxt(modelname + '_test_idx.txt')
    
    
    
    
    st_file = modelname +'_stat_input.txt'
    file1 = open(st_file, 'r') 
    active_st_idx = []
    clon = []
    clat = []
    for i, line in enumerate(file1):
        pieces = line.split()
        if np.int(pieces[6]): # This is an active station
            active_st_idx.append(i) #Save its index
        clon.append(np.float(pieces[3]))
        clat.append(np.float(pieces[4]))
    
    file1.close()
    
    active_st_idx = np.array(active_st_idx)
    
    
    # Load test data from database
    # Data parameters
    t_max = 315
    n_stations = len(active_st_idx)
    ncomp = 3
    

    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")
    # device = torch.device('cpu')
    dist.barrier()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % idr_torch.local_rank}
    model = PEGSNet(ncomp,n_stations, t_max).to(gpu)
    model = DDP(model,device_ids=[idr_torch.local_rank]).to(gpu)
    model.module.load_state_dict(torch.load(modelname +'.pth', map_location=map_location))
    
    model.double()
    model.eval()
    
    batch_size_per_gpu = 200
    
    # Parameters
    params = {'batch_size': batch_size_per_gpu,
              'shuffle': False,
              'num_workers': 40,
              'pin_memory':True}

    
    test_idx = np.sort(test_idx)
    #indici = [19,8,16,0,1,14,2,7]  
    indici = np.arange(0,len(test_idx),dtype=int)
    test_indici = test_idx[indici]
    
    
    
    # with h5py.File(db_name, 'r') as f:
    #         #    label =  np.array(f['label'][ind])  
    #         y_Mw = np.array(f['eq_params'][np.sort(test_idx),0])  
            
    
    # Each process (GPU) will save tot_samples/4 examples. So divide the lenght of test set by four
    samp_per_gpu = int(len(indici)/4)
    TRUE = np.zeros((samp_per_gpu, t_max+1,4))
    PREDS = np.zeros((samp_per_gpu, t_max+1,4))
    MW = np.zeros(samp_per_gpu)
    NOISE =  np.zeros((samp_per_gpu, n_stations))   
    SNR =  np.zeros((samp_per_gpu, t_max+1))
    
    
    # For each time step make predictions on a batch
    for i in range(t_max+1):
        if idr_torch.rank ==0:
            print('Processing time_step: ' + str(i) + '/' +str(t_max+1))
            # Generators
        test_set = HDF5Dataset(test_indici, 
                                   n_stations, 
                                   t_max, 
                                   ncomp,
                                   i,
                                   database_path=db_name
                                   )


        test_sampler = DistributedEvalSampler(test_set, num_replicas=idr_torch.size,
                                                                    rank=idr_torch.rank)


        test_loader = torch.utils.data.DataLoader(test_set, **params, sampler=test_sampler)

        
        #test_loader = torch.utils.data.DataLoader(test_set, **params)
        c = 0
        with torch.no_grad():
            for j, (data, y, Mw,ava_stat, NOISE0, STF, index) in enumerate(test_loader):
                

                test_data = data.to(gpu, non_blocking=True)
                res = model(test_data.double())
                cpu_pred = res.cpu()
                result = cpu_pred.data.numpy()
                NOISE0 = NOISE0.numpy()
                start = j*batch_size_per_gpu
                end = start + NOISE0.shape[0]

                # It's working ok. Saving tot_samples/4 per gpu
                #print('RANK: {} -- first index {} / last index {}'.format(idr_torch.rank, index[0], index[-1]))
                #print('RANK: {} -- {}  {}'.format(idr_torch.rank, start,end))
                #if idr_torch.rank == 0:
                    #print (start,end)
                    #print (index[:3])
                    #if i==0:
                        #print(NOISE0.shape, c, j)
                
                if i==0:
                    
                    for kk in range(NOISE0.shape[0]):
                        
                        NOISE[c,:] = np.std(NOISE0[kk,:,:350,0], axis =1)
                        c = c+1
                    MW[start:end] = Mw.numpy()
        
    
    
                # Rescale predictions
                minmw=5.5
                maxmw=10.0
                minlat = -43.0
                maxlat = -22.0
                # maxlat = 45.3
                minlon = -74.6
                maxlon = -70.3
                # maxlon = 150.7
               
                # Rescale if normalization was [0,1] 
                # mw_pred = (result[:,0] *(maxmw - minmw))  +minmw
                # lat_pred = (result[:,1]*(maxlat-minlat)) + minlat
                # lon_pred = (result[:,2] * (maxlon-minlon)) + minlon

                # REscale if normalization was [-1,1]
                mw_pred = ( ((1.0 + result[:,0])/2.0) *(maxmw - minmw) ) +minmw
                lat_pred = ( ((1.0 + result[:,1])/2.0) *(maxlat - minlat) ) + minlat
                lon_pred = ( ((1.0 + result[:,2])/2.0) *(maxlon - minlon) )+ minlon
               
                # Rescale if normalization was just max value
                #lat_pred = result[:,1]*maxlat
                #lon_pred = result[:,2]*maxlon
                #mw_pred = result[:,0]*maxmw
            

                TRUE[start:end,i,0] = y.numpy()[:,0]
                TRUE[start:end,i,1] = y.numpy()[:,1]
                TRUE[start:end,i,2] = y.numpy()[:,2]
                TRUE[start:end,i,3] = STF.numpy()
                PREDS[start:end,i,0] = mw_pred
                PREDS[start:end,i,1] = lat_pred
                PREDS[start:end,i,2] = lon_pred
                PREDS[start:end,i,3] = ava_stat.numpy()
                    
        
            
    np.save(modelname+'_MW_test_rank{}.npy'.format(idr_torch.rank),MW) 
    np.save(modelname+'_PREDS_test_rank{}.npy'.format(idr_torch.rank),PREDS)
    np.save(modelname+'_TRUE_test_rank{}.npy'.format(idr_torch.rank),TRUE)
    np.save(modelname+'_NOISE_test_rank{}.npy'.format(idr_torch.rank),NOISE)
    np.save(modelname+'_SNR_test_rank{}.npy'.format(idr_torch.rank),SNR)
            
    cleanup()
if __name__ == "__main__":

    # parse arguments
    args = parse_args()
    if args is None:
        print("No args found. Exit...")
        exit()

    main(args)
 
