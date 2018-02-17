#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/types.h>
#include "mpi-help.h"
#include "cuda-help.h"
#include "tca-help.h"

#define COUNT 1024*512
#define WARMUP 10
#define TIMES 500
