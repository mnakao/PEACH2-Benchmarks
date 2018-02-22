#include "common.h"
void mpi_allgather(const int count, const int my_rank, const int num_procs, const int output_flag);

int main(int argc, char **argv)
{
  int my_rank, num_procs;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  CUDA_SAFE_CALL(cudaSetDevice(0));

  mpi_allgather(2, my_rank, num_procs, 0);
  for(int count=2; count<=COUNT; count*=2)
    mpi_allgather(count, my_rank, num_procs, 1);
  
  MPI_Finalize();

  return 0;
}

static void mpi_allgather(const int count, const int my_rank, const int num_procs, const int output_flag)
{
  double start_time, my_time, sum_time;
  size_t send_byte  = count * sizeof(double);
  size_t whole_byte = count * sizeof(double) * num_procs;

  double *host_send_array,  *device_send_array;
  double *host_whole_array, *device_whole_array;
  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_send_array,  send_byte));
  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_whole_array, whole_byte));
  CUDA_SAFE_CALL(cudaMalloc((void**)&device_send_array,  send_byte));
  CUDA_SAFE_CALL(cudaMalloc((void**)&device_whole_array, whole_byte));

  for(int i=0; i<count; i++)
    host_send_array[i] = (double)my_rank;
  
  CUDA_SAFE_CALL(cudaMemcpy(device_send_array, host_send_array, send_byte, cudaMemcpyDefault));
  MPI_Barrier(MPI_COMM_WORLD);
  sum_time = 0.0;
  for(int t=0; t<WARMUP+TIMES; t++){
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(t >= WARMUP)
      start_time = MPI_Wtime();
    
    MPI_Allgather(device_send_array, count, MPI_DOUBLE, device_whole_array, count, MPI_DOUBLE, MPI_COMM_WORLD);
    
    if (t >= WARMUP) {
      my_time = MPI_Wtime() - start_time;
      MPI_Allreduce(MPI_IN_PLACE, &my_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      sum_time += my_time;
    }
  }
  
  if(my_rank == 0 && output_flag == 1)
    printf("size = %zu %lf us\n", send_byte, (sum_time/TIMES)*1e6);

  CUDA_SAFE_CALL(cudaMemcpy(host_whole_array, device_whole_array, whole_byte, cudaMemcpyDefault));
  
  for(int i=0; i<num_procs; i++)
    for(int j=0; j<count; j++)
      if(fabs(host_whole_array[i*count+j] - (double)i) > 1e-16){
	printf("Incorrect answer.\n");
	break;
      }
  
  CUDA_SAFE_CALL(cudaFreeHost(host_send_array));
  CUDA_SAFE_CALL(cudaFreeHost(host_whole_array));
  CUDA_SAFE_CALL(cudaFree(device_send_array));
  CUDA_SAFE_CALL(cudaFree(device_whole_array));
}
