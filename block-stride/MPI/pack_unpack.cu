#include <cuda.h>
#define IDXV(x, y, ld)   ((x) + (y) * (ld))
#define block 128
#define grid  256

__global__ static void pack_matrix(double* __restrict__ device_cube,
                                   double* __restrict__ tmp_matrix,
                                   const int row, const int column, const int depth)
{
  //  for(int i=0; i<row; i++)
  //    for(int k=0; k<depth; k++)
  //      tmp_matrix[i][k] = device_cube[i][0][k];
  //      tmp_matrix[i*N+k] = device_cube[i*N*N+k];

  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);
  while(ivx < row*depth){
    int i = ivx / row;
    int k = ivx - i * row;
    tmp_matrix[ivx] = device_cube[i*column*depth + k];
    ivx += blockDim.x * gridDim.x;
  }
}

__global__ static void unpack_matrix(double* __restrict__ tmp_matrix,
                                     double* __restrict__ device_cube,
                                     const int row, const int column, const int depth)
{
  //  for(int i=0; i<row; i++)
  //    for(int k=0; k<depth; k++)
  //      device_cube[i][column-1][k] = tmp_matrix[i][k];
  ////    device_cube[i*N*N+(column-1)*N+k] = tmp_matrix[i*N+k];

  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);
  while(ivx < row*depth){
    int i = ivx / row;
    int k = ivx - i * row;
    device_cube[i*column*depth+(column-1)*depth+k] = tmp_matrix[ivx];
    ivx += blockDim.x * gridDim.x;
  }
}

extern "C"
void call_pack(double* __restrict__ device_cube,
	       double* __restrict__ tmp_matrix,
	       const int row, const int column, const int depth)
{
  pack_matrix <<< grid, block >>> (device_cube, tmp_matrix, row, column, depth);
  cudaDeviceSynchronize();
}

extern "C"
void call_unpack(double* __restrict__ tmp_matrix,
		 double* __restrict__ device_cube,
		 const int row, const int column, const int depth)
{
  unpack_matrix <<< grid, block >>> (tmp_matrix, device_cube, row, column, depth);
  cudaDeviceSynchronize();
}
