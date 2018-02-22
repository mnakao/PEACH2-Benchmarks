#include "common.h"

int main(int argc, char **argv)
{
  int my_rank, nprocs;
  size_t byte;
  double start_time, my_time, sum_time;
  double *host_buffer, *device_send_buffer, *device_recv_buffer;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  CUDA_SAFE_CALL(cudaSetDevice(0));

  for(int count=1; count<=COUNT; count*=2){
    byte = count * sizeof(double);

    CUDA_SAFE_CALL(cudaMallocHost((void**)&host_buffer, byte));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_send_buffer, byte));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_recv_buffer, byte));

    for(int i=0; i<count; i++)
      host_buffer[i] = (double)my_rank;

    CUDA_SAFE_CALL(cudaMemcpy(device_send_buffer, host_buffer, byte, cudaMemcpyDefault));
    MPI_Barrier(MPI_COMM_WORLD);
    sum_time = 0.0;
    for(int t=0; t<WARMUP+TIMES; t++){
      MPI_Barrier(MPI_COMM_WORLD);

      if(t >= WARMUP)
    	start_time = MPI_Wtime();
      
      MPI_Allreduce(device_send_buffer, device_recv_buffer, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      if (t >= WARMUP) {
        my_time = MPI_Wtime() - start_time;
        MPI_Allreduce(MPI_IN_PLACE, &my_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        sum_time += my_time;
      }
    }

    CUDA_SAFE_CALL(cudaMemcpy(host_buffer, device_recv_buffer, byte, cudaMemcpyDefault));
    
    for(int i=0; i<count; i++)
      if(fabs(host_buffer[i] - (double)((nprocs)*(nprocs-1)/2)) > 1e16)
	printf("failed\n");
	 
    if(my_rank == 0)
      printf("size = %zu %lf us\n", byte, (sum_time/TIMES)*1e6);

    CUDA_SAFE_CALL(cudaFreeHost(host_buffer));
    CUDA_SAFE_CALL(cudaFree(device_send_buffer));
    CUDA_SAFE_CALL(cudaFree(device_recv_buffer));
  }
  
  MPI_Finalize();

  return 0;
}
