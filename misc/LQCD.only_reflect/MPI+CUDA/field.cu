/*
   Linear algebra.
                             [2016.05.03 Hideo Matsufuru]
*/
#include "lattice.h"

static __global__ void dot_dev(real_t __restrict__ *a, const real_t* __restrict__ v1, const real_t* __restrict__ v2)
{
  __shared__ real_t cache[VECTOR_LENGTH];
  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);
  int cacheIndex = threadIdx.x;

  real_t tmp = 0;
  while (ivx < (LT2-2)*(LZ2-2)*yx_Spinor) {
    int it = ivx / ((LZ2-2)*yx_Spinor);
    int iz = (ivx - it*((LZ2-2)*yx_Spinor)) / yx_Spinor;
    int iyx = ivx % yx_Spinor;
    int ivx2 = (it+1)*LZ2*yx_Spinor + (iz+1)*yx_Spinor + iyx;

    tmp += v1[ivx2] * v2[ivx2];
    ivx += blockDim.x * gridDim.x;
  }

  cache[cacheIndex] = tmp;
  __syncthreads();

  int i = blockDim.x/2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }

  if(cacheIndex == 0)
    a[blockIdx.x] = cache[0];
}

real_t dot(const real_t* __restrict__ v1, const real_t* __restrict__ v2)
{
  static real_t tmp[NUM_GANGS], *tmp_dev;
  static int STATIC_FLAG = 1;

  if(STATIC_FLAG) HANDLE_ERROR( cudaMalloc((void**)&tmp_dev, NUM_GANGS*sizeof(real_t)) );
  dot_dev <<< NUM_GANGS, VECTOR_LENGTH >>> (tmp_dev, v1, v2);
  HANDLE_ERROR( cudaMemcpy(tmp, tmp_dev, NUM_GANGS*sizeof(real_t), cudaMemcpyDeviceToHost) );
  
  real_t a = 0.0;
  for (int i=0; i<NUM_GANGS; i++)
    a += tmp[i];

  MPI_Allreduce(MPI_IN_PLACE, &a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  STATIC_FLAG = 0;
  return a;
}

static __global__ void norm2_t_dev(real_t __restrict__ *a, const real_t* __restrict__ v1, const int len)
{
  __shared__ real_t cache[VECTOR_LENGTH];
  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);
  int cacheIndex = threadIdx.x;

  real_t tmp = 0.0;
  while (ivx < len) {
    tmp += v1[ivx] * v1[ivx];
    ivx += blockDim.x * gridDim.x;
  }

  cache[cacheIndex] = tmp;
  __syncthreads();

  int i = blockDim.x/2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }

  if(cacheIndex == 0)
    a[blockIdx.x] = cache[0];
}

void norm2_t(real_t* __restrict__ corr, const real_t* __restrict__ v1)
{
  static real_t tmp[NUM_GANGS], *tmp_dev;
  static int STATIC_FLAG = 1;

  if(STATIC_FLAG) HANDLE_ERROR( cudaMalloc((void**)&tmp_dev, NUM_GANGS*sizeof(real_t)) );

  for(int it=1; it<LT2-1; it++){
    norm2_t_dev <<< NUM_GANGS, VECTOR_LENGTH >>> (tmp_dev, v1+(it*LZ2+1)*yx_Spinor, (LZ2-2)*yx_Spinor);
    HANDLE_ERROR( cudaMemcpy(tmp, tmp_dev, NUM_GANGS*sizeof(real_t), cudaMemcpyDeviceToHost) );

    real_t a = 0;
    for (int i=0; i<NUM_GANGS; i++)
      a += tmp[i];

    corr[it-1] += a;
  }

  STATIC_FLAG = 0;
}

static __global__ void norm2_dev(real_t __restrict__ *a, const real_t* __restrict__ v1, const int len)
{
  __shared__ real_t cache[VECTOR_LENGTH];
  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);
  int cacheIndex = threadIdx.x;

  real_t tmp = 0;
  while (ivx < len) {
    int it = ivx / ((LZ2-2)*yx_Spinor);
    int iz = (ivx - it*((LZ2-2)*yx_Spinor)) / yx_Spinor;
    int iyx = ivx % yx_Spinor;
    int ivx2 = (it+1)*LZ2*yx_Spinor + (iz+1)*yx_Spinor + iyx;

    tmp += v1[ivx2] * v1[ivx2];
    ivx += blockDim.x * gridDim.x;
  }

  cache[cacheIndex] = tmp;
  __syncthreads();

  int i = blockDim.x/2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }

  if(cacheIndex == 0)
    a[blockIdx.x] = cache[0];
}

real_t norm2(const real_t* __restrict__ v1)
{
  static real_t tmp[NUM_GANGS], *tmp_dev;
  static int STATIC_FLAG = 1;

  if(STATIC_FLAG) HANDLE_ERROR( cudaMalloc((void**)&tmp_dev, NUM_GANGS*sizeof(real_t)) );
  norm2_dev <<< NUM_GANGS, VECTOR_LENGTH >>> (tmp_dev, v1, (LT2-2)*(LZ2-2)*yx_Spinor);
  HANDLE_ERROR( cudaMemcpy(tmp, tmp_dev, NUM_GANGS*sizeof(real_t), cudaMemcpyDeviceToHost) );

  real_t a = 0;
  for (int i=0; i<NUM_GANGS; i++)
    a += tmp[i];

  MPI_Allreduce(MPI_IN_PLACE, &a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  STATIC_FLAG = 0;
  return a;
}

static __global__ void axpy_dev(real_t* __restrict__ v, const real_t a, const real_t* __restrict__ w)
{
  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);

  while (ivx < (LT2-2)*(LZ2-2)*yx_Spinor) {
    int it = ivx / ((LZ2-2)*yx_Spinor);
    int iz = (ivx - it*((LZ2-2)*yx_Spinor)) / yx_Spinor;
    int iyx = ivx % yx_Spinor;
    int ivx2 = (it+1)*LZ2*yx_Spinor + (iz+1)*yx_Spinor + iyx;
    
    v[ivx2] += a * w[ivx2];
    ivx += blockDim.x * gridDim.x;
  }
}

void axpy(real_t* __restrict__ v, const real_t a, const real_t* __restrict__ w)
{
  axpy_dev <<< NUM_GANGS, VECTOR_LENGTH >>> (v, a, w);
}

static __global__ void scal_dev(real_t* __restrict__ v, const real_t a)
{
  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);

  while (ivx < (LT2-2)*(LZ2-2)*yx_Spinor) {
    int it = ivx / ((LZ2-2)*yx_Spinor);
    int iz = (ivx - it*((LZ2-2)*yx_Spinor)) / yx_Spinor;
    int iyx = ivx % yx_Spinor;
    int ivx2 = (it+1)*LZ2*yx_Spinor + (iz+1)*yx_Spinor + iyx;

    v[ivx2] *= a;
    ivx += blockDim.x * gridDim.x;
  }
}

void scal(real_t* __restrict__ v, const real_t a)
{
  scal_dev <<< NUM_GANGS, VECTOR_LENGTH >>> (v, a);
}

static __global__ void copy_dev(real_t* __restrict__ v, const real_t* __restrict__ w)
{
  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);

  while (ivx < (LT2-2)*(LZ2-2)*yx_Spinor) {
    int it = ivx / ((LZ2-2)*yx_Spinor);
    int iz = (ivx - it*((LZ2-2)*yx_Spinor)) / yx_Spinor;
    int iyx = ivx % yx_Spinor;
    int ivx2 = (it+1)*LZ2*yx_Spinor + (iz+1)*yx_Spinor + iyx;
    
    v[ivx2] = w[ivx2];
    ivx += blockDim.x * gridDim.x;
  }
}

void copy(real_t* __restrict__ v, const real_t* __restrict__ w)
{
  copy_dev <<< NUM_GANGS, VECTOR_LENGTH >>> (v, w);
}
