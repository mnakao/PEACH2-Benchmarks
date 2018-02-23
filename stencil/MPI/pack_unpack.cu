#include <cuda.h>
#define IDXV(x, y, ld)   ((x) + (y) * (ld))
#define block 128
#define grid  256

__global__ static void pack_matrix(double* __restrict__ device_cube,
                                   double* __restrict__ tmp_matrix,
                                   const int n, const int index)
{
  //  for(int z=0; z<n; z++)
  //    for(int x=0; x<n; x++)
  //      tmp_matrix[z][x]  = device_cube[z][index][x];
  //      tmp_matrix[z*n+x] = device_cube[z*n*n+index*n+x];

  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);
  while(ivx < n*n){
    int z = ivx / n;
    int x = ivx - z * n;
    tmp_matrix[ivx] = device_cube[z*n*n+index*n+x];
    ivx += blockDim.x * gridDim.x;
  }
}

__global__ static void unpack_matrix(double* __restrict__ tmp_matrix,
                                     double* __restrict__ device_cube,
				     const int n, const int index)
{
  //  for(int z=0; z<n; z++)
  //    for(int x=0; x<n; x++)
  //      device_cube[z][index][x] = tmp_matrix[z][x];
  //      device_cube[z*n*n+index*n+x] = tmp_matrix[z*n+x];

  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);
  while(ivx < n*n){
    int z = ivx / n;
    int x = ivx - z * n;
    device_cube[z*n*n+index*n+x] = tmp_matrix[ivx];
    ivx += blockDim.x * gridDim.x;
  }
}

extern "C"
void call_pack(double* __restrict__ device_cube,
	       double* __restrict__ tmp_matrix,
	       const int n, const int index)
{
  pack_matrix <<< grid, block >>> (device_cube, tmp_matrix, n, index);
  cudaDeviceSynchronize();
}

extern "C"
void call_unpack(double* __restrict__ tmp_matrix,
		 double* __restrict__ device_cube,
		 const int n, const int index)
{
  unpack_matrix <<< grid, block >>> (tmp_matrix, device_cube, n, index);
  cudaDeviceSynchronize();
}
