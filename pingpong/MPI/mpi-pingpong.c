#include "common.h"

static void pingpong(const int count, const int my_rank, const int output_flag)
{
  int target = (my_rank + 1) % 2;
  int transfer_byte = count * sizeof(double);
  double start, end;
  double *host_buf, *dummy_buf;
  double *device_sendbuf, *device_recvbuf;

  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_buf,  transfer_byte));
  CUDA_SAFE_CALL(cudaMallocHost((void**)&dummy_buf, transfer_byte));
  CUDA_SAFE_CALL(cudaMalloc((void**)&device_sendbuf, transfer_byte));
  CUDA_SAFE_CALL(cudaMalloc((void**)&device_recvbuf, transfer_byte));

  for(int i=0; i<count; i++){
    host_buf[i]  = (double)((my_rank+1)*i);
    dummy_buf[i] = -99.0;
  }

  CUDA_SAFE_CALL(cudaMemcpy(device_sendbuf, host_buf,  transfer_byte, cudaMemcpyDefault));
  CUDA_SAFE_CALL(cudaMemcpy(device_recvbuf, dummy_buf, transfer_byte, cudaMemcpyDefault));

  for(int t=0; t<TIMES+WARMUP; t++) {
    if(t == WARMUP){
      MPI_Barrier(MPI_COMM_WORLD);
      start = MPI_Wtime();
    }
    if(my_rank == 0){
      MPI_SAFE_CALL(MPI_Send(device_sendbuf, count, MPI_DOUBLE, target, 0, MPI_COMM_WORLD));
      MPI_SAFE_CALL(MPI_Recv(device_recvbuf, count, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    else{
      MPI_SAFE_CALL(MPI_Recv(device_recvbuf, count, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      MPI_SAFE_CALL(MPI_Send(device_sendbuf, count, MPI_DOUBLE, target, 0, MPI_COMM_WORLD));
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();

  CUDA_SAFE_CALL(cudaMemcpy(host_buf, device_recvbuf, transfer_byte, cudaMemcpyDefault));
  for(int i=0; i<count; i++)
    if(fabs(host_buf[i] - (double)(target+1)*i) > 1e-18){
      printf("ERROR %f %f %d %d\n", host_buf[i], (double)(target+1)*i, i, my_rank);
      break;
    }

  double one_way_comm_time = ((end-start)/TIMES/2)*1e6;
  double bandwidth = transfer_byte / one_way_comm_time;
  if(my_rank == 0 && output_flag)
    printf("size = %d one_way_comm_time = %lf [usec], bandwidth = %lf [MB/s]\n", transfer_byte, one_way_comm_time, bandwidth);
  
  CUDA_SAFE_CALL(cudaFreeHost(host_buf));
  CUDA_SAFE_CALL(cudaFreeHost(dummy_buf));
  CUDA_SAFE_CALL(cudaFree(device_sendbuf));
  CUDA_SAFE_CALL(cudaFree(device_recvbuf));
}

int main(int argc, char** argv)
{
  int my_rank, namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_SAFE_CALL(MPI_Init(&argc, &argv));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
  MPI_Get_processor_name(processor_name, &namelen);
  fprintf(stdout,"Process %d on %s\n", my_rank, processor_name);

  CUDA_SAFE_CALL(cudaSetDevice(0));

  pingpong(1, my_rank, 0); // Dry run
  for(int count=1; count<=COUNT; count*=2)
    pingpong(count, my_rank, 1);

  MPI_SAFE_CALL(MPI_Finalize());

  return 0;
}
