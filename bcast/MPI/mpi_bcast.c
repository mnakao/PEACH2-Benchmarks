#include "common.h"

int main(int argc, char **argv)
{
  int my_rank;
  size_t byte;
  double start_time, my_time, sum_time;
  double *host_buffer, *device_buffer;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  CUDA_SAFE_CALL(cudaSetDevice(0));

  for(int count=1; count<=COUNT; count*=2){
    byte = count * sizeof(double);

    CUDA_SAFE_CALL(cudaMallocHost((void**)&host_buffer, byte));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_buffer, byte));

    if(my_rank == 0)
      for(int i=0; i<count; i++)
	host_buffer[i] = (double)i;

    CUDA_SAFE_CALL(cudaMemcpy(device_buffer, host_buffer, byte, cudaMemcpyDefault));
    MPI_Barrier(MPI_COMM_WORLD);
    sum_time = 0.0;
    for(int t=0; t<WARMUP+TIMES; t++){
      MPI_Barrier(MPI_COMM_WORLD);

      if(t >= WARMUP)
    	start_time = MPI_Wtime();
      
      MPI_Bcast(device_buffer, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      if (t >= WARMUP) {
        my_time = MPI_Wtime() - start_time;
        MPI_Allreduce(MPI_IN_PLACE, &my_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        sum_time += my_time;
      }
    }

    CUDA_SAFE_CALL(cudaMemcpy(host_buffer, device_buffer, byte, cudaMemcpyDefault));
    
    for(int i=0; i<count; i++)
      if(fabs(host_buffer[i] - (double)i) > 1e16)
	printf("failed\n");
	 
    if(my_rank == 0)
      printf("size = %zu %lf us\n", byte, (sum_time/TIMES)*1e6);

    CUDA_SAFE_CALL(cudaFreeHost(host_buffer));
    CUDA_SAFE_CALL(cudaFree(device_buffer));
  }
  
  MPI_Finalize();

  return 0;
}
