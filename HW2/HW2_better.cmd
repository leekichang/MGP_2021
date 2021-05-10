####################
##
## Test Condor command file
##
####################

executable	= HTtest
output		= better.out
error		= better.err
request_cpus = 16
log		= better.log
arguments	= 5000 64 500000 10 1
queue
