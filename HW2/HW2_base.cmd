####################
##
## Test Condor command file
##
####################

executable	= HTtest
 output		= base.out
 error		= base.err
 request_cpus = 16
 log		= base.log
 arguments	= 5000 64 500000 10 0
 queue
