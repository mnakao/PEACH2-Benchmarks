#include <cuda.h>
#define IDXV(x, y, ld)   ((x) + (y) * (ld))
#define block 128
#define grid  256

__global__ static void pack_matrix(double* __restrict__ device_cube,
                                   double* __restrict__ tmp_matrix,
                                   const int n)
{
  //  for(int i=0; i<n; i++)
  //    for(int k=0; k<n; k++)
  //      tmp_matrix[i][k] = device_cube[i][0][k];
  //      tmp_matrix[i*N+k] = device_cube[i*N*N+k];

  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);
  while(ivx < n*n){
    int i = ivx / n;
    int k = ivx - i * n;
    tmp_matrix[ivx] = device_cube[i*n*n + k];
    ivx += blockDim.x * gridDim.x;
  }
}

__global__ static void unpack_matrix(double* __restrict__ tmp_matrix,
                                     double* __restrict__ device_cube,
                                     const int n)
{
  //  for(int i=0; i<n; i++)
  //    for(int k=0; k<n; k++)
  //      device_cube[i][n-1][k] = tmp_matrix[i][k];
  ////    device_cube[i*N*N+(n-1)*N+k] = tmp_matrix[i*N+k];

  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);
  while(ivx < n*n){
    int i = ivx / n;
    int k = ivx - i * n;
    device_cube[i*n*n+(n-1)*n+k] = tmp_matrix[ivx];
    ivx += blockDim.x * gridDim.x;
  }
}

extern "C"
void call_pack(double* __restrict__ device_cube,
	       double* __restrict__ tmp_matrix,
	       const int n)
{
  pack_matrix <<< grid, block >>> (device_cube, tmp_matrix, n);
  cudaDeviceSynchronize();
}

extern "C"
void call_unpack(double* __restrict__ tmp_matrix,
		 double* __restrict__ device_cube,
		 const int n)
{
  unpack_matrix <<< grid, block >>> (tmp_matrix, device_cube, n);
  cudaDeviceSynchronize();
}
