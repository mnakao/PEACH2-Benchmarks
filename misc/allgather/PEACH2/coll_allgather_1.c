#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "tca-api.h"
#include <cuda_runtime.h>

void tcaCreateHandleList(tcaHandle **handle, int num_proc, double *ptr, size_t byte);

#define CUDA_SAFE_CALL(cuda_call) do {                                  \
    cudaError_t status = cuda_call;                                     \
    if(status != cudaSuccess) {                                         \
      fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",     \
	      __FILE__, __LINE__, cudaGetErrorString(status) );         \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

#define TCA_SAFE_CALL(call) do {					\
    tcaresult result = call;						\
    if (result != TCA_SUCCESS) {					\
      fprintf(stderr, "TCA error in file '%s' in line %i : %s.\n",	\
	      __FILE__, __LINE__, tcaGetErrorString(result));		\
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)

#define COUNT 2
#define WARMUP 0
#define TIMES 1

int main (int argc, char **argv)
{
  int my_rank, num_proc;
  int i, t, count;
  static size_t whole_byte, byte;
  static double start, end, my_time, max_time, sum_time;

  double *host_send_buffer, *host_recv_buffer, *host_ans_recv_buffer;
  double *device_send_buffer, *device_recv_buffer, *device_ans_recv_buffer;

  tcaHandle send_handle, *handles;
  int mask, target_rank;
  tcaDesc *tca_desc;
  int const dmac_ch = 0;
  int const wait_tag = 0x100;

  MPI_Init(&argc, &argv);
  TCA_SAFE_CALL(tcaInit());
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

  CUDA_SAFE_CALL(cudaSetDevice(0));

  for (count = 2; count <= COUNT; count *= 2) {
    byte = count * sizeof(double);
    whole_byte = byte * num_proc;

    CUDA_SAFE_CALL(cudaMallocHost((void**)&host_send_buffer, byte));
    CUDA_SAFE_CALL(cudaMallocHost((void**)&host_recv_buffer, whole_byte));
    CUDA_SAFE_CALL(cudaMallocHost((void**)&host_ans_recv_buffer, whole_byte));
    TCA_SAFE_CALL(tcaMalloc((void**)&device_send_buffer, byte, tcaMemoryGPU));
    TCA_SAFE_CALL(tcaMalloc((void**)&device_recv_buffer, whole_byte, tcaMemoryGPU));
    TCA_SAFE_CALL(tcaMalloc((void**)&device_ans_recv_buffer, whole_byte, tcaMemoryGPU));

    for (i = 0; i < count; i++) {
      host_send_buffer[i] = (double)my_rank;
    }

    for (i = 0; i < (count * num_proc); i++) {
      host_recv_buffer[i] = -1.0;
      host_ans_recv_buffer[i] = -1.0;
    }

    CUDA_SAFE_CALL(cudaMemcpy(device_send_buffer, host_send_buffer, byte, cudaMemcpyDefault));
    CUDA_SAFE_CALL(cudaMemcpy(device_recv_buffer, host_recv_buffer, whole_byte, cudaMemcpyDefault));
    CUDA_SAFE_CALL(cudaMemcpy(device_ans_recv_buffer, host_ans_recv_buffer, whole_byte, cudaMemcpyDefault));

    // Init Coll Allgather
    tca_desc = tcaDescNew();
    TCA_SAFE_CALL(tcaCreateHandle(&send_handle, device_send_buffer, byte, tcaMemoryGPU));

    tcaCreateHandleList(&handles, num_proc, device_recv_buffer, whole_byte);

    int dmaFlag = tcaDMAUseInternal|tcaDMAUseNotifyInternal|tcaDMANotify;

    int rank_group = my_rank;

    TCA_SAFE_CALL(tcaDescSetMemcpy(tca_desc,
				   &handles[my_rank], (rank_group * byte),
				   &send_handle, 0,
				   byte, dmaFlag, 0, wait_tag));

    for (i = 1, mask = 1; mask < 2; mask <<= 1, i++) { // mask : 1 -> 2 -> 4 -> 8 -> 16 -> ...
      target_rank = my_rank ^ mask;
      
      off_t offset = rank_group * byte;

      /* printf("[%d] : mask = %d, target_rank = %d, index = %d, offset = %ld, wait_slot = %d\n", my_rank, mask, target_rank, rank_group, offset, i); */

      TCA_SAFE_CALL(tcaDescSetMemcpy(tca_desc,
				     &handles[target_rank], offset,
				     &handles[my_rank], offset,
				     (mask * byte), dmaFlag, i, wait_tag));

      if (my_rank >= target_rank) {
        rank_group -= mask;
      }
    }

    sum_time = 0.0;
    for (t = 0; t < (WARMUP + TIMES); t++) {
      MPI_Barrier(MPI_COMM_WORLD);

      if (t >= WARMUP)
    	start = MPI_Wtime();

      // Insert Communication
      /* MPI_Allgather(device_send_buffer, count, MPI_DOUBLE, device_ans_recv_buffer, count, MPI_DOUBLE, MPI_COMM_WORLD); */

      TCA_SAFE_CALL(tcaDescSet(tca_desc, dmac_ch));
      TCA_SAFE_CALL(tcaStartDMADesc(dmac_ch));
      TCA_SAFE_CALL(tcaWaitDMARecvDesc(&send_handle, 0, wait_tag));

      for (i = 1, mask = 1; mask < 2; mask <<= 1, i++) { // mask : 1 -> 2 -> 4 -> 8 -> 16 -> ...
      	target_rank = my_rank ^ mask;
      	TCA_SAFE_CALL(tcaWaitDMARecvDesc(&handles[target_rank], i, wait_tag));
      }

      if (t >= WARMUP) {
    	end = MPI_Wtime();
    	my_time = (end - start);
    	MPI_Allreduce(&my_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    	sum_time += max_time;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    CUDA_SAFE_CALL(cudaMemcpy(host_recv_buffer, device_recv_buffer, whole_byte, cudaMemcpyDefault));
    /* CUDA_SAFE_CALL(cudaMemcpy(host_ans_recv_buffer, device_ans_recv_buffer, whole_byte, cudaMemcpyDefault)); */

    if (my_rank == 1) {
      printf("%lf\n", (sum_time / TIMES)*1e6);
      for (i = 0; i < (count * num_proc); i++) {
      	printf("%lf\t", host_recv_buffer[i]);
      }
      printf("\n");
    }

    free(handles);

    CUDA_SAFE_CALL(cudaFreeHost(host_send_buffer));
    CUDA_SAFE_CALL(cudaFreeHost(host_recv_buffer));
    CUDA_SAFE_CALL(cudaFreeHost(host_ans_recv_buffer));
    TCA_SAFE_CALL(tcaFree(device_send_buffer, tcaMemoryGPU));
    TCA_SAFE_CALL(tcaFree(device_recv_buffer, tcaMemoryGPU));
    TCA_SAFE_CALL(tcaFree(device_ans_recv_buffer, tcaMemoryGPU));

    TCA_SAFE_CALL(tcaDescFree(tca_desc));
  }

  
  MPI_Finalize();

  return 0;
}
