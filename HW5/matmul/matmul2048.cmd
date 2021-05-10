############################################
##
## Matrix Multiplication Condor command file
##
############################################

executable	 = bin/matmul
output		 = result/matmul.out
error		 = result/matmul.err
log		     = result/matmul.log
environment = "LD_LIBRARY_PATH=/usr/local/cuda/lib64"
request_cpus = 16
should_transfer_files   = YES 
when_to_transfer_output = ON_EXIT
transfer_input_files    = data/input_2048.txt, data/output_2048.txt
arguments	            = input_2048.txt output_2048.txt
queue
