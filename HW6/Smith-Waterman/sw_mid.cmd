############################################
##
## PageRank Condor command file
##
############################################

executable	 = bin/sw
output		 = result/sw.out
error		 = result/sw.err
log		     = result/sw.log
environment = "LD_LIBRARY_PATH=/usr/local/cuda/lib64"
request_cpus = 16
should_transfer_files   = YES 
when_to_transfer_output = ON_EXIT
transfer_input_files    = /nfs/home/mgp2021_data/sw/target_mid.fasta, /nfs/home/mgp2021_data/sw/query_mid.fasta
arguments	            = -p -t query_mid.fasta target_mid.fasta
queue
