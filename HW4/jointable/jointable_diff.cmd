############################################
##
## Join Tables Condor command file
##
############################################

executable	 = jointable
output		 = result/jointable.out
error		 = result/jointable.err
log		     = result/jointable.log
request_cpus = 16
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
##Change below two lines to test your generated input
##transfer_input_files    = data/input.bin, data/output.bin                                                            
arguments	            = /nfs/home/mgp2021_data/join_data/input_10m50m.bin /nfs/home/mgp2021_data/join_data/output_10m50m.bin 0 
queue