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
transfer_input_files    = data/input_4096.txt, data/output_4096.txt
arguments	            = input_4096.txt output_4096.txt
queue
