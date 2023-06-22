# PEGSNet_Chile
#
# TRAINING 
# The training process requires to run the file run_train_model_torch_hotshot.py by specifying the following parameters:
# --db_name: name of database without the extension
# --train_nsamp: number of samples for training
# --val_nsamp: number of samples for validation
# --test_nsamp: number of samples for testing
# --batch_size: size of the batches (we used 512) 
# --lr: learning rate (we used 0.001)
# --n_epochs: number of epochs (we used 100)
# --log_inteval: interval batches (we used 100) 
# --ncomp: number of components (we used 3)
# --model_outname: name of the output model
#
# TESTING
# Testing the already trained model requires to run the file run_predict_all_parallel_DDP.py by specifying the parameters:
# --model_name: name of the model obtained in the previous step
# --data_path: the database on wich the model is to be tested

# Files Classes.py and Classes_predict.py contain the needed functions to run "run_train_model_torch_hotshot.py" and 
# "run_predict_all_parallel_DDP.py", respectively.