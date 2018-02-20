#include "common.h"
void mpi_bcast(const int count, const int my_rank, const int output_flag);

int main(int argc, char **argv)
{
  int my_rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  CUDA_SAFE_CALL(cudaSetDevice(0));

  for(int count=1; count<=COUNT; count*=2)
    mpi_bcast(count, my_rank, 1);
  
  MPI_Finalize();

  return 0;
}

static void mpi_bcast(const int count, const int my_rank, const int output_flag)
{
  double start_time, my_time, sum_time;
  double *host_array, *device_array;

  size_t byte = count * sizeof(double);
  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_array, byte));
  CUDA_SAFE_CALL(cudaMalloc((void**)&device_array, byte));

  if(my_rank == 0) for(int i=0; i<count; i++) host_array[i] = (double)i;
  else             for(int i=0; i<count; i++) host_array[i] = -1.0;

  CUDA_SAFE_CALL(cudaMemcpy(device_array, host_array, byte, cudaMemcpyDefault));
  MPI_Barrier(MPI_COMM_WORLD);
  sum_time = 0.0;
  for(int t=0; t<WARMUP+TIMES; t++){
    MPI_Barrier(MPI_COMM_WORLD);

    if(t >= WARMUP)
      start_time = MPI_Wtime();

    MPI_Bcast(device_array, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (t >= WARMUP) {
      my_time = MPI_Wtime() - start_time;
      MPI_Allreduce(MPI_IN_PLACE, &my_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      sum_time += my_time;
    }
  }

  CUDA_SAFE_CALL(cudaMemcpy(host_array, device_array, byte, cudaMemcpyDefault));

  for(int i=0; i<count; i++)
    if(fabs(host_array[i] - (double)i) > 1e16)
      printf("failed\n");

  if(my_rank == 0 && output_flag == 1)
    printf("size = %zu %lf us\n", byte, (sum_time/TIMES)*1e6);

  CUDA_SAFE_CALL(cudaFreeHost(host_array));
  CUDA_SAFE_CALL(cudaFree(device_array));
 }
