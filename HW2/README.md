# HW2 of MGP

## Summary

You will implement hash table with striped lock.

The `HT` have three functions.

- `insert`
- `contains`
- `remove`

## Contents

### Expected Result

```bash
$ make remote_base
condor_submit HW2_base.cmd
Submitting job(s).
1 job(s) submitted to cluster 43784.

$ make remote_better
condor_submit HW2_better.cmd
Submitting job(s).
1 job(s) submitted to cluster 43785.

$ make report
cat base.out
start filling
init hash table took 0.00377415 sec
start test
test 5000000 ops took 1.43904 sec
Sanity check: there are 5000 items left
cat better.out
start filling
init hash table took 0.00174668 sec
start test
test 5000000 ops took !@$%!@$% sec
Sanity check: there are 5000 items left
```

### Make Functions

```bash
# Run locally
make
make run
# Compile
make HTtest
# Clean
make clean
# Request base job to condor remote server
make remote_base
# Request better job to condor remote server
make remote_better
# Check queue of condor
make queue
# Print output of outs
```

## References

- [std::thread - cplusplus.com](http://www.cplusplus.com/reference/thread/thread/)
- [OpenMP 4.5 C/C++ Syntax Guide](https://www.openmp.org/wp-content/uploads/OpenMP-4.5-1115-CPP-web.pdf)
- [HTCondor commands cheat-sheet](https://raggleton.github.io/condor-cheatsheet/)
- [CMake Cheatsheet](https://github.com/mortennobel/CMake-Cheatsheet/blob/master/CMake_Cheatsheet.pdf)
