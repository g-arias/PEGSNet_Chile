#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:24:14 2020

@author: andrea
"""

from datetime import datetime
from time import time
import os, shutil, argparse,sys
import torch
from torch import  optim
from Classes import HDF5Dataset, PEGSNet, DistributedEvalSampler
import random
import numpy as np
# from torchsummary import summary
# from datetime import datetime
# from torch.nn import functional as F

# from torchvision.utils import save_image

from torch import nn
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
    desc = "Calculate statistics of a given dataset"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--db_name', type=str, default='../DATABASES/run_db_test',                   
                        help='Path of the preprocessed database (without .hdf5)')
    
    
    parser.add_argument('--model_outname', type=str, default='test',                   
                        help='Prefix name for output model. A subdirectory under ./outputs will be created)')
    
    parser.add_argument('--train_nsamp', type=int, default=500, 
                        help='Number of examples for the training set.')
    
    parser.add_argument('--val_nsamp', type=int, default=50, 
                        help='Number of examples for the validation set.') 
    
    parser.add_argument('--test_nsamp', type=int, default=50, 
                        help='Number of examples for the test set.')  
    
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='The size of the batches.')  

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')

    parser.add_argument('--ncomp', type=int, default=3,
                        help='Number of seismic components to use.')
    
    parser.add_argument('--n_epochs', type=int, default=100, 
                        help='Number of epochs.')  
    
    
    parser.add_argument('--log_interval', type=int, default=100, 
                        help='Print info every log_interval batches.') 
    
    parser.add_argument('--check_interval', type=int, default=5,
                        help='Save a checkpoint every check_interval epochs.')


    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    #   --PEGS_db_path       
    if not os.path.exists(args.db_name+'.hdf5'):
        print(args.db_name + '.hdf5 does not exist')
        exit()


    return args


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
          nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight)
      nn.init.constant_(m.bias, 0)

# def enable_dropout(model):
#     """ Function to enable the dropdistriout layers during test-time """
#     for m in model.modules():
#         if m.__class__.__name__.startswith('Dropout'):
#             print("switching dropout on " + str(m.__class__.__name__))
#             m.train()    

    


    
def main(args):
    
    
    
    print("Running PEGSNET_STF (DDP) on rank {}.".format(idr_torch.rank))
    
    if idr_torch.rank==0:
        print("ME, PROCESS {}, I am MASTER".format(idr_torch.rank))
    
    setup()


    working_db = args.db_name
    modelprefix = args.model_outname
    ##################  Parameters of train/val/test sets ####################
    # Specifying these paraemters allows for smaller train/val/test for testing the algo
    # I have changed the way the index for train/val/test are used. Now they are fixed 
    # for each database and are loaded from ../DATABASES/run_db??_train_idx.npy
    train_nsamp = args.train_nsamp
    val_nsamp = args.val_nsamp
    test_nsamp = args.test_nsamp
    tot_samp = train_nsamp + val_nsamp + test_nsamp



    batch_size = args.batch_size
    batch_size_per_gpu = batch_size // idr_torch.size

    lr = args.lr

    max_epochs = args.n_epochs
    log_interval = args.log_interval
    check_interval = args.check_interval


    #Load index for train/val/test
    # Only select the firs n_samples based on the values specified as input 
    list_train = np.load(working_db +"_train_idx.npy")[:train_nsamp]
    list_val = np.load(working_db +"_val_idx.npy")[:val_nsamp]
    list_test = np.load(working_db +"_test_idx.npy")[:test_nsamp]

    if idr_torch.rank ==  0:
        # Do some checks on dimensions and number of samples as specified in input parameters
        if (train_nsamp > len(list_train)):
            print (" train_nsamp is bigger than the maximum number of training samples for this database which is {}".format(len(list_train)))
            print (" Please reduce the value of train_nsamp")
            sys.exit("Exit")
        if (val_nsamp > len(list_val)):
            print (" val_nsamp is bigger than the maximum number of validation samples for this database which is {}".format(len(list_val)))
            print (" Please reduce the value of val_nsamp")
            sys.exit("Exit")
        if (test_nsamp > len(list_test)):
            print (" test_nsamp is bigger than the maximum number of testing samples for this database which is {}".format(len(list_test)))
            print (" Please reduce the value of test_nsamp")
            sys.exit("Exit")

    
    ################### CREATE OUTPUT DIRECTORY #########################################
    modeldir = "./outputs/model_"+ modelprefix
    

    modelname = modeldir + "/model_" + modelprefix + '_' + str(int(train_nsamp/1000)) + "k" + str(int(val_nsamp/1000)) + "k"
    
    if idr_torch.rank ==0:
        if not os.path.exists(modeldir):
            os.makedirs(modeldir) 
    
    
    
    # LOAD ACTIVE stations index and sort lon/lat 
    # USELESS now. The database has active stations already sorted
    st_file = './stations/stat_input.txt'
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
    # Data parameters
    t_max = 315
    n_stations = len(active_st_idx)
    n_comp = args.ncomp
    

    # CREATE MODEL
    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")
    model0 = PEGSNet(n_comp,n_stations, t_max).to(gpu)
    model0 = model0.double()
    model0.apply(initialize_weights)
    Pmodel = DDP(model0,device_ids=[idr_torch.local_rank])
    # Select the optimizer with learning rate
    optimizer = optim.Adam(Pmodel.parameters(), lr=lr)

###############   CHOOSE LOSS function #############
# 1) MSE
#    criterion = torch.nn.MSELoss()
# 2) MAE
    criterion = torch.nn.L1Loss(reduction='mean')
# 3) Huber
#    criterion = torch.nn.SmoothL1Loss()
    
    
    
#############  SPECIFY training parameters    
    # Parameters
    params = {'batch_size': batch_size_per_gpu,
              'shuffle': False,
              'num_workers': 40,
              'pin_memory':True}
    
    
    # Datasets
    
    partition = { 
        "train": list_train,
        "validation": list_val,
        "test" : list_test
        }# IDs
    
    
#  The generator will load the data     
    # Generators
    training_set = HDF5Dataset(partition['train'], 
                               n_stations, 
                               t_max, 
                               n_comp,
                               database_path=working_db+".hdf5"
                               )
    
# Sampler is needed by DistributedDataParallel
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_set,
                                                                    num_replicas=idr_torch.size,
                                                                    rank=idr_torch.rank)
    
# The loader will load the data using generator and sampler
    train_loader = torch.utils.data.DataLoader(training_set, **params, sampler=train_sampler)
    
    
    
   
# Same for validation set. The only thing that changes is the sampler that takes care of uneven batches. 
    validation_set = HDF5Dataset(partition['validation'], 
                                 n_stations, 
                                 t_max, 
                                 n_comp,
                                 database_path=working_db+".hdf5"                                
                                 )
    
    
    val_sampler = DistributedEvalSampler(validation_set, num_replicas=idr_torch.size,
                                                                    rank=idr_torch.rank)
    
    
    
    val_loader = torch.utils.data.DataLoader(validation_set, **params, sampler=val_sampler)



        
        
########   START TRAINING LOOP 
    train_losses = []
    val_losses = []
    if idr_torch.rank == 0: start = datetime.now() 
# This is looping over all epochs
    for epoch in range(1, max_epochs + 1):
        if idr_torch.rank == 0:
            start_dataload = time()
        # Set epoch is needed to avoid deterministic sampling 
        train_sampler.set_epoch(epoch)
        
        # Set the model in train mode
        Pmodel.train()
   
        train_loss = 0.0
        
        # Start looping over batches for the training dataset
        for batch_idx, (data, y , index) in enumerate(train_loader):

            if idr_torch.rank == 0: stop_dataload = time()
            
            # Load data and labels on GPU
            train_labels = y.to(gpu, non_blocking=True)
            train_data = data.to(gpu, non_blocking=True)

            if idr_torch.rank == 0: start_training = time()
            
            # if PROCESS_RANK == 0 or PROCESS_RANK == 1:
            #     print (f"RANK {PROCESS_RANK} -- EPOCH{epoch} -- BATCH {batch_idx}")
            #     print (index)
             
            
            # Predict labels using current batch
            train_pred = Pmodel(train_data.double())
            # Evaluate the loss for this batch
            loss = criterion(train_pred,train_labels)
            #Loss is the mean over all examples in this batch. Default behaviour of criterion.
            train_loss += loss.item()
            
            # Do backprop
            optimizer.zero_grad()
            loss.backward()
            # Take a step of optimizer
            optimizer.step()


    #         print (" RANK: {} -- batchidx:{} -- \
    #                len(train_loader) {} -- \
    #                    len(dataset) {} -- len(index) {} \
    #                        -- loss {:.2f} -- train_loss {:.2f}".format(
    # PROCESS_RANK, batch_idx, 
    # len(train_loader), 
    # len(train_loader.dataset), 
    # len(index), loss.item(), train_loss))


            if idr_torch.rank == 0: stop_training = time()
            
            # Print some infos after log_interval steps for this epoch
            if batch_idx % log_interval == 0 and idr_torch.rank ==  0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Time data load: {:.3f}ms, Time training: {:.3f}ms'.format(
                    epoch, batch_idx * len(data)*idr_torch.size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(), (stop_dataload - start_dataload)*1000, (stop_training - start_training)*1000 ))
                # print(len(data))
                if idr_torch.rank == 0: start_dataload = time()

        nbatches = batch_idx+1
        # Save train loss for this epoch
        train_losses.append(train_loss/nbatches)

        # CHeckpoint interval is not implemented yet.
        #if  idr_torch.rank==0 and epoch % check_interval:
            
        #    torch.save(Pmodel.module.state_dict(), modelname +"_CHECK_{}.pth".format(epoch))
        
        if idr_torch.rank ==  0 or idr_torch.rank==1:
            print('====> Epoch: {} Average loss: {:.4f} -- nbatches: {}'.format(
                  epoch, train_loss / nbatches, nbatches))
        
    
        # Once the epoch is over check on the validation set   
        val_sampler.set_epoch(epoch)
        # Set model in eval mode but leave dropout active
        Pmodel.eval()
        # Activate dropout layers
        for m in Pmodel.modules():
            if m.__class__.__name__.startswith('Dropout'):
                # print("switching dropout on " + str(m.__class__.__name__))
                m.train() 
        test_loss = 0
        # No gradients are computed here.
        with torch.no_grad():
            if idr_torch.rank ==0: val_start_dataload = time()
            
            # Start looping over batches of the validation set
            for i, (data, y, index) in enumerate(val_loader):
                if idr_torch.rank == 0: val_stop_dataload = time()
                
                # Load data and labels on GPU
                test_labels = y.to(gpu, non_blocking=True)
                test_data = data.to(gpu, non_blocking=True)

                if idr_torch.rank == 0: start_testing = time()
                test_pred = Pmodel(test_data.double())
                tloss = criterion(test_pred,test_labels)
                test_loss += tloss.item()

                if idr_torch.rank == 0: stop_testing = time()

                # Print some infos on the validation set 
                if i % log_interval == 0 and idr_torch.rank ==  0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t val loss: {:.6f} Time val data load: {:.3f}ms, Time testing: {:.3f}ms'.format(
                    epoch, i * len(data)*idr_torch.size, len(val_loader.dataset),
                    100. * i / len(val_loader),
                    tloss.item(), (val_stop_dataload - val_start_dataload)*1000, (stop_testing - start_testing)*1000 ))
                    
                    if idr_torch.rank == 0: val_start_dataload = time()



    #             # nsamples += len(index)
    #             print (" RANK: {} -- batchidx:{} -- \
    #                 len(val_loader) {} -- \
    #                     len(dataset) {} -- len(index) {} \
    #                         -- loss {:.2f} -- train_loss {:.2f}".format(
    # PROCESS_RANK, i, 
    # len(val_loader), 
    # len(val_loader.dataset), 
    # len(index), tloss.item(), test_loss))
                
                
                
        nbatches = i+1       
        test_loss /= nbatches
        val_losses.append(test_loss)
        if idr_torch.rank ==  0:
            print('====> Test set loss: {:.4f} -- nbatches {}'.format(test_loss, nbatches))
        
        

        
        
    ########### CHECK HOW LOSSES ARE SAVED ############
    # We might need to calculate exact losses as these are just 
    # mean losses for each batch. This can be biased if batches are of 
    # different sizes. 
    # Check Here:
    # https://discuss.pytorch.org/t/plotting-loss-curve/42632/3

    # For the moment save lossess for each rank. These will be averaged later
    np.savetxt(modelname +  "RANK_{}_train_losses.txt".format(idr_torch.rank), np.array(train_losses))   
    np.savetxt(modelname +  "RANK_{}_val_losses.txt".format(idr_torch.rank), np.array(val_losses))  
        
    # MASTER NODE will save some stuff and copy some files in output folder
    if idr_torch.rank==0:
        print(">>> Training complete in: " + str(datetime.now() - start))
        torch.save(Pmodel.module.state_dict(), modelname +".pth")

        #Save list of index for test set (any of these have been used in train+val)
        np.savetxt(modelname + "_test_idx.txt", list_test)
        print (modelname  + "_test_idx.txt has been saved in folder " + modeldir)  
        
        np.savetxt(modelname  + "_val_idx.txt", list_val)
        print (modelname + "_val_idx.txt has been saved in folder " + modeldir) 
       
        
        shutil.copy2(st_file, modelname +"_stat_input.txt")
        shutil.copy2("Classes.py", modelname +"_Classes.py")
        shutil.copy2("./run_script.sh", modelname +"_run_script.sh")
        
        
    cleanup()

    

if __name__ == '__main__':    
    # parse arguments
    args = parse_args()
    if args is None:
        print("No args found. Exit...")
        exit()
    
    main(args)   
        
    
        
























