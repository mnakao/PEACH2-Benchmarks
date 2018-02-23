#include "common.h"
extern void call_pack(double* __restrict__, double* __restrict__, const int);
extern void call_unpack(double* __restrict__, double* __restrict__, const int);

void verify(double *host, int n, int my_rank)
{
  for(int i=0; i<n; i++) 
    for(int j=0; j<n-1; j++) 
      for(int k=0; k<n; k++)
	if(fabs(host[i*(n*n)+j*n+k] - (double)((my_rank+1)*(i*(n*n)+j*n+k))) > 1e-18)
	  printf("Error1\n");

  int target = (my_rank+1)%2;  
  for(int i=0; i<n; i++)
    for(int k=0; k<n; k++)
      if(fabs(host[i*(n*n)+(n-1)*n+k] - (double)((target+1)*(i*(n*n)+k))) > 1e-18)
	printf("Error2 [%d] host[%d[%d][%d] %f != %f\n", 
	       my_rank, i, n-1, k, host[i*(n*n)+(n-1)*n+k], (double)((target+1)*(i*(n*n)+k)));
}

static void block_stride(const int n, const int my_rank, const int output_flag)
{
  int target = (my_rank + 1) % 2;
  size_t cube_byte   = n * n * n * sizeof(double);
  size_t matrix_byte = n * n * sizeof(double);
  double *host_cube, *device_cube, *tmp_matrix;
  double start, end;
  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_cube, cube_byte));
  CUDA_SAFE_CALL(cudaMalloc((void**)&device_cube, cube_byte));
  CUDA_SAFE_CALL(cudaMalloc((void**)&tmp_matrix, matrix_byte));

  for(int i=0; i<n; i++)
    for(int j=0; j<n; j++)
      for(int k=0; k<n; k++)
	host_cube[i*(n*n)+j*n+k] = (double)((my_rank+1)*(i*(n*n)+j*n+k));

  CUDA_SAFE_CALL(cudaMemcpy(device_cube, host_cube, cube_byte, cudaMemcpyDefault));
  MPI_Barrier(MPI_COMM_WORLD);

  for(int t=0; t<TIMES+WARMUP; t++){
    if(t == WARMUP){
      MPI_Barrier(MPI_COMM_WORLD);
      start = MPI_Wtime();
    }

    if(my_rank == 0){
      call_pack(device_cube, tmp_matrix, n);
      MPI_SAFE_CALL(MPI_Send(tmp_matrix, n*n, MPI_DOUBLE, target, 0, MPI_COMM_WORLD));

      MPI_SAFE_CALL(MPI_Recv(tmp_matrix, n*n, MPI_DOUBLE, target, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      call_unpack(tmp_matrix, device_cube, n);
    }
    else{
      MPI_SAFE_CALL(MPI_Recv(tmp_matrix, n*n, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      call_unpack(tmp_matrix, device_cube, n);

      call_pack(device_cube, tmp_matrix, n);
      MPI_SAFE_CALL(MPI_Send(tmp_matrix, n*n, MPI_DOUBLE, target, 1, MPI_COMM_WORLD));
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();
  CUDA_SAFE_CALL(cudaMemcpy(host_cube, device_cube, cube_byte, cudaMemcpyDefault));
  verify(host_cube, n, n, n, my_rank);

  double one_way_comm_time = ((end - start)/TIMES/2)*1e6;
  double bandwidth         = matrix_byte / one_way_comm_time;
  if(my_rank == 0 && output_flag == 1)
    printf("N = %d one_way_comm_time = %lf [usec], bandwidth = %lf [MB/s]\n", count, one_way_comm_time, bandwidth);

  CUDA_SAFE_CALL(cudaFreeHost(host_cube));
  CUDA_SAFE_CALL(cudaFree(device_cube));
  CUDA_SAFE_CALL(cudaFree(tmp_matrix));
}

int main(int argc, char** argv)
{
  int my_rank; 
  MPI_SAFE_CALL(MPI_Init(&argc, &argv));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
  CUDA_SAFE_CALL(cudaSetDevice(0));

  block_stride(2, my_rank, 0); // Dry run
  for(int count=2; count<=COUNT; count*=2)
    block_stride(count, my_rank, 1);
  
  MPI_SAFE_CALL(MPI_Finalize());
  
  return 0;
}
