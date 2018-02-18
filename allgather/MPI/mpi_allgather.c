#include "common.h"

int main (int argc, char **argv)
{
  int my_rank, num_proc;
  size_t whole_byte, byte;
  double start, end;
  double *host_send_buffer, *host_recv_buffer;
  double *device_send_buffer, *device_recv_buffer;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  CUDA_SAFE_CALL(cudaSetDevice(0));

  for(int count=2; count<=COUNT; count*=2){
    byte       = count * sizeof(double);
    whole_byte = byte * num_proc;

    CUDA_SAFE_CALL(cudaMallocHost((void**)&host_send_buffer, byte));
    CUDA_SAFE_CALL(cudaMallocHost((void**)&host_recv_buffer, whole_byte));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_send_buffer, byte));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_recv_buffer, whole_byte));

    for(int i=0; i<count; i++)
      host_send_buffer[i] = (double)my_rank;

    CUDA_SAFE_CALL(cudaMemcpy(device_send_buffer, host_send_buffer, byte, cudaMemcpyDefault));
    MPI_Barrier(MPI_COMM_WORLD);

    for(int t=0; t<WARMUP+TIMES; t++){
      if(t == WARMUP){
	MPI_Barrier(MPI_COMM_WORLD);
    	start = MPI_Wtime();
      }
      
      MPI_Allgather(device_send_buffer, count, MPI_DOUBLE, device_recv_buffer, count, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    CUDA_SAFE_CALL(cudaMemcpy(host_recv_buffer, device_recv_buffer, whole_byte, cudaMemcpyDefault));

    for(int i=0; i<num_proc; i++)
      for(int j=0; j<count; j++)
	if(fabs(host_recv_buffer[i*count+j] - (double)i) > 1e16)
	  printf("failed\n");
    
    if(my_rank == 0)
      printf("size = %zu %lf us\n", byte, ((end-start)/TIMES)*1e6);

    CUDA_SAFE_CALL(cudaFreeHost(host_send_buffer));
    CUDA_SAFE_CALL(cudaFreeHost(host_recv_buffer));
    CUDA_SAFE_CALL(cudaFree(device_send_buffer));
    CUDA_SAFE_CALL(cudaFree(device_recv_buffer));
  }
  
  MPI_Finalize();

  return 0;
}
