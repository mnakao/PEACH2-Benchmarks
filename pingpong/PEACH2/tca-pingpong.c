#include "common.h"

int main(int argc, char** argv)
{
  int transfer_byte, my_rank, other;
  double start, end;
  double *host_buf, *dummy_buf;
  double *device_sendbuf, *device_recvbuf;
  tcaHandle *sendbuf_handle, *recvbuf_handle;
  tcaDesc *tca_desc;
  int const dma_ch = 0;
  int const wait_tag = 0x100;

  MPI_Init(&argc, &argv);
  TCA_SAFE_CALL(tcaInit());
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  other = (my_rank+1)%2;
  CUDA_SAFE_CALL(cudaSetDevice(0));

  for(int count=1; count<=COUNT; count*=2){
    transfer_byte = count * sizeof(double);

    CUDA_SAFE_CALL(cudaMallocHost((void**)&host_buf,  transfer_byte));
    CUDA_SAFE_CALL(cudaMallocHost((void**)&dummy_buf, transfer_byte));

    TCA_SAFE_CALL(tcaMalloc((void**)&device_sendbuf, transfer_byte, tcaMemoryGPU));
    TCA_SAFE_CALL(tcaMalloc((void**)&device_recvbuf, transfer_byte, tcaMemoryGPU));
    
    for(int i = 0; i < count; i++) {
      host_buf[i]  = (double)((my_rank+1)*i);
      dummy_buf[i] = -99.0;
    }

    CUDA_SAFE_CALL(cudaMemcpy(device_sendbuf, host_buf, transfer_byte, cudaMemcpyDefault));
    CUDA_SAFE_CALL(cudaMemcpy(device_recvbuf, dummy_buf, transfer_byte, cudaMemcpyDefault));

    tcaCreateHandleList(&sendbuf_handle, 2, device_sendbuf, transfer_byte);
    tcaCreateHandleList(&recvbuf_handle, 2, device_recvbuf, transfer_byte);

    tca_desc = tcaDescNew();

    const int dmaFlag = tcaDMAUseInternal|tcaDMAUseNotifyInternal|tcaDMANotify|tcaDMANotifySelf;
    TCA_SAFE_CALL(tcaDescSetMemcpy(tca_desc, &recvbuf_handle[other], 0, &sendbuf_handle[my_rank], 0, transfer_byte, dmaFlag, 0, wait_tag));
    TCA_SAFE_CALL(tcaDescSet(tca_desc, dma_ch));
    MPI_Barrier(MPI_COMM_WORLD);

    for(int t=0; t<TIMES+WARMUP; t++){
      if(t == WARMUP) {
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
      }
      if(my_rank == 0){
	TCA_SAFE_CALL(tcaStartDMADesc(dma_ch));
	TCA_SAFE_CALL(tcaWaitDMARecvDesc(&sendbuf_handle[my_rank], 0, wait_tag));
	TCA_SAFE_CALL(tcaWaitDMARecvDesc(&recvbuf_handle[other], 0, wait_tag));
      } 
      else{
	TCA_SAFE_CALL(tcaWaitDMARecvDesc(&recvbuf_handle[other], 0, wait_tag));
	TCA_SAFE_CALL(tcaStartDMADesc(dma_ch));
	TCA_SAFE_CALL(tcaWaitDMARecvDesc(&sendbuf_handle[my_rank], 0, wait_tag));
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    CUDA_SAFE_CALL(cudaMemcpy(host_buf, device_recvbuf, transfer_byte, cudaMemcpyDefault));
    for(int i=0; i<count; i++)
      if(fabs(host_buf[i] - (double)((other+1)*i)) > 1e-18)
	printf("ERROR @ %d, count = %d\n", my_rank, count);
    
    double latency = ((end-start)/TIMES/2)*1e6;
    double bandwidth = transfer_byte / latency;
    if(my_rank == 0) 
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
  
  MPI_Finalize();

  return 0;
}
