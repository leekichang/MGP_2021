############################################
##
## PageRank Condor command file
##
############################################

executable	 = bin/pr
output		 = result/pr.out
error		 = result/pr.err
log		     = result/pr.log
environment = "LD_LIBRARY_PATH=/usr/local/cuda/lib64"
request_cpus = 16
should_transfer_files   = YES 
when_to_transfer_output = ON_EXIT
transfer_input_files    = /nfs/home/mgp2021_data/pagerank/livejournal.el, /nfs/home/mgp2021_data/pagerank/livejournal_answer.txt
arguments	            = -f livejournal.el -c livejournal_answer.txt -k 50
queue
