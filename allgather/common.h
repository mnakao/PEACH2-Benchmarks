#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <sys/types.h> 
#include <unistd.h>
#include "tca-help.h"
#include "cuda-help.h"

#define COUNT 1024*64
#define WARMUP 10
#define TIMES 100
