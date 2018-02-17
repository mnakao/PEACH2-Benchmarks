#include "common.h"

int main(int argc, char** argv)
{
  int transfer_byte, my_rank, other;
  double start, end;
  double *host_array, *device_send_array, *device_recv_array;

  MPI_SAFE_CALL(MPI_Init(&argc, &argv));
  TCA_SAFE_CALL(tcaInit());
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));

  other = (my_rank + 1) % 2;
  CUDA_SAFE_CALL(cudaSetDevice(0));

  for(int count=1; count<=COUNT; count*=2) {
    transfer_byte = count * sizeof(double);
    CUDA_SAFE_CALL(cudaMallocHost((void**)&host_array, transfer_byte));

    for(int i=0; i<count; i++)
      host_array[i] = (double)((my_rank+1)*i);

    TCA_SAFE_CALL(tcaMalloc((void**)&device_send_array, transfer_byte, tcaMemoryGPU));
    TCA_SAFE_CALL(tcaMalloc((void**)&device_recv_array, transfer_byte, tcaMemoryGPU));

    CUDA_SAFE_CALL(cudaMemcpy(device_send_array, host_array, transfer_byte, cudaMemcpyDefault));
    CUDA_SAFE_CALL(cudaMemcpy(device_recv_array, host_array, transfer_byte, cudaMemcpyDefault));

    MPI_Barrier(MPI_COMM_WORLD);

    for(int t=0; t<TIMES+WARMUP; t++) {
      if(t == WARMUP){
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
      }
      if(my_rank == 0){
	MPI_SAFE_CALL(MPI_Send(device_send_array, count, MPI_DOUBLE, other, 0, MPI_COMM_WORLD));
	MPI_SAFE_CALL(MPI_Recv(device_recv_array, count, MPI_DOUBLE, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      } 
      else{
	MPI_SAFE_CALL(MPI_Recv(device_recv_array, count, MPI_DOUBLE, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
	MPI_SAFE_CALL(MPI_Send(device_send_array, count, MPI_DOUBLE, other, 0, MPI_COMM_WORLD));
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    CUDA_SAFE_CALL(cudaMemcpy(host_array, device_recv_array, transfer_byte, cudaMemcpyDefault));
    for(int i=0; i<count; i++){
      if(fabs(host_array[i] - (double)(other+1)*i) > 1e-18){
	printf("ERROR %f %f %d %d\n", host_array[i], (double)(other+1)*i, i, my_rank);
	//	MPI_SAFE_CALL(MPI_Finalize());
	//	exit(1);
      }
    }

    double one_way_comm_time = ((end-start)/TIMES/2)*1e6;
    double bandwidth = transfer_byte / one_way_comm_time;
    if(my_rank == 0)
      printf("size = %d one_way_comm_time = %lf [usec], bandwidth = %lf [MB/s]\n", transfer_byte, one_way_comm_time, bandwidth);
    
    CUDA_SAFE_CALL(cudaFreeHost(host_array));
    TCA_SAFE_CALL(tcaFree(device_send_array, tcaMemoryGPU));
    TCA_SAFE_CALL(tcaFree(device_recv_array, tcaMemoryGPU));
  }

  MPI_SAFE_CALL(MPI_Finalize());

  return 0;
}
