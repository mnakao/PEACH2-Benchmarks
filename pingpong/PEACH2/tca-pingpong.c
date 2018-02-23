#include "common.h"
#define DMA_CH 0
#define WAIT_TAG (0x100)
#define DMA_FLAG (tcaDMAUseInternal|tcaDMAUseNotifyInternal|tcaDMANotify|tcaDMANotifySelf)

static void pingpong(const int count, const int my_rank, const int output_flag)
{
  int target = (my_rank + 1) % 2;
  int transfer_byte = count * sizeof(double);
  double start, end;
  double *host_buf, *dummy_buf;
  double *device_sendbuf, *device_recvbuf;
  tcaHandle *sendbuf_handle, *recvbuf_handle;
  tcaDesc *tca_desc;

  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_buf,  transfer_byte));
  CUDA_SAFE_CALL(cudaMallocHost((void**)&dummy_buf, transfer_byte));
  TCA_SAFE_CALL(tcaMalloc((void**)&device_sendbuf, transfer_byte, tcaMemoryGPU));
  TCA_SAFE_CALL(tcaMalloc((void**)&device_recvbuf, transfer_byte, tcaMemoryGPU));

  for(int i=0; i<count; i++) {
    host_buf[i]  = (double)((my_rank+1)*i);
    dummy_buf[i] = -99.0;
  }

  CUDA_SAFE_CALL(cudaMemcpy(device_sendbuf, host_buf, transfer_byte,  cudaMemcpyDefault));
  CUDA_SAFE_CALL(cudaMemcpy(device_recvbuf, dummy_buf, transfer_byte, cudaMemcpyDefault));

  tcaCreateHandleList(&sendbuf_handle, 2, device_sendbuf, transfer_byte);
  tcaCreateHandleList(&recvbuf_handle, 2, device_recvbuf, transfer_byte);

  tca_desc = tcaDescNew();
  TCA_SAFE_CALL(tcaDescSetMemcpy(tca_desc, &recvbuf_handle[target], 0, &sendbuf_handle[my_rank], 0, transfer_byte, DMA_FLAG, 0, WAIT_TAG));
  TCA_SAFE_CALL(tcaDescSet(tca_desc, DMA_CH));
  MPI_Barrier(MPI_COMM_WORLD);

  for(int t=0; t<TIMES+WARMUP; t++){
    if(t == WARMUP) {
      MPI_Barrier(MPI_COMM_WORLD);
      start = MPI_Wtime();
    }
    if(my_rank == 0){
      TCA_SAFE_CALL(tcaStartDMADesc(DMA_CH));
      TCA_SAFE_CALL(tcaWaitDMARecvDesc(&sendbuf_handle[my_rank], 0, WAIT_TAG));
      TCA_SAFE_CALL(tcaWaitDMARecvDesc(&recvbuf_handle[target],  0, WAIT_TAG));
    }
    else{
      TCA_SAFE_CALL(tcaWaitDMARecvDesc(&recvbuf_handle[target],  0, WAIT_TAG));
      TCA_SAFE_CALL(tcaStartDMADesc(DMA_CH));
      TCA_SAFE_CALL(tcaWaitDMARecvDesc(&sendbuf_handle[my_rank], 0, WAIT_TAG));
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();

  CUDA_SAFE_CALL(cudaMemcpy(host_buf, device_recvbuf, transfer_byte, cudaMemcpyDefault));
  for(int i=0; i<count; i++)
    if(fabs(host_buf[i] - (double)((target+1)*i)) > 1e-18){
      printf("ERROR @ %d, count = %d\n", my_rank, count);
      break;
    }

  double latency   = ((end-start)/TIMES/2)*1e6;
  double bandwidth = transfer_byte / latency;
  if(my_rank == 0 && output_flag == 1)
    printf("data size = %d, latency = %lf[usec], bandwidth = %lf[MB/s]\n", transfer_byte, latency, bandwidth);

  MPI_Barrier(MPI_COMM_WORLD);
  free(sendbuf_handle);
  free(recvbuf_handle);
  CUDA_SAFE_CALL(cudaFreeHost(host_buf));
  CUDA_SAFE_CALL(cudaFreeHost(dummy_buf));
  TCA_SAFE_CALL(tcaFree(device_sendbuf, tcaMemoryGPU));
  TCA_SAFE_CALL(tcaFree(device_recvbuf, tcaMemoryGPU));
  TCA_SAFE_CALL(tcaDescFree(tca_desc));
}

int main(int argc, char** argv)
{
  int my_rank;

  MPI_SAFE_CALL(MPI_Init(&argc, &argv));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
  TCA_SAFE_CALL(tcaInit());
  TCA_SAFE_CALL(  tcaReset());
  MPI_SAFE_CALL(MPI_Finalize());  return 0;
  CUDA_SAFE_CALL(cudaSetDevice(0));

  pingpong(1, my_rank, 0); // Dry run
  for(int count=1; count<=COUNT; count*=2)
    pingpong(count, my_rank, 1);
  
  MPI_SAFE_CALL(MPI_Finalize());

  return 0;
}
