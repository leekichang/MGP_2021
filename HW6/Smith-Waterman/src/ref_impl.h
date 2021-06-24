#include <iostream>
#include <math.h>
#include <omp.h>

#include "args_parser.h" // include iostream, string, fstream
#include "dataset.h"
#include "smith_waterman_parallel.h"
#include "similarity_algorithm_parallel.h"

void mgp_opt_align(Parameters *args, Dataset *dataset);



