#include "common.h"
#define DMA_CH 0
#define WAIT_TAG (0x100)
#define DMA_FLAG (tcaDMAUseInternal|tcaDMAUseNotifyInternal|tcaDMANotify|tcaDMANotifySelf)

void verify(double *host, int n, int my_rank)
{
  if(my_rank == 1){
    for(int z=0; z<n-1; z++)
      for(int y=0; y<n; y++)
        for(int x=0; x<n; x++)
          if(fabs(host[z*(n*n)+y*n+x] - (double)(my_rank+1)) > 1e-16)
            printf("Error1\n");

    for(int y=0; y<n; y++)
      for(int x=0; x<n; x++)
	if(fabs(host[(n-1)*(n*n)+y*n+x] - 1.0) > 1e-16)
	  printf("Error2\n");
  }
  else if(my_rank == 3){
    for(int z=1; z<n; z++)
      for(int y=0; y<n; y++)
        for(int x=0; x<n; x++)
          if(fabs(host[z*(n*n)+y*n+x] - (double)(my_rank+1)) > 1e-16)
            printf("[%d] Error1\n", my_rank);

    for(int y=0; y<n; y++)
      for(int x=0; x<n; x++)
        if(fabs(host[y*n+x] - 1.0) > 1e-16)
	  printf("[%d] Error2\n", my_rank);
  }
  else if(my_rank == 2){
    for(int z=0; z<n; z++)
      for(int y=1; y<n; y++)
        for(int x=0; x<n; x++)
          if(fabs(host[z*(n*n)+y*n+x] - (double)(my_rank+1)) > 1e-16)
	    printf("[%d] Error1\n", my_rank);

    for(int z=0; z<n; z++)
      for(int x=0; x<n; x++)
        if(fabs(host[z*(n*n)+x] - 1.0) > 1e-16)
	    printf("[%d] Error2\n", my_rank);
  }
  else if(my_rank == 4){
    for(int z=0; z<n; z++)
      for(int y=0; y<n-1; y++)
        for(int x=0; x<n; x++)
          if(fabs(host[z*(n*n)+y*n+x] - (double)(my_rank+1)) > 1e-16)
	    printf("[%d] Error1\n", my_rank);

    for(int z=0; z<n; z++)
      for(int x=0; x<n; x++)
        if(fabs(host[z*(n*n)+(n-1)*n+x] - 1.0) > 1e-16)
	  printf("[%d] Error2\n", my_rank);
  }
}

static void stencil(const int n, const int my_rank, const int output_flag)
{
  size_t cube_byte   = (n * n * n) * sizeof(double);
  size_t matrix_byte = (n * n) * sizeof(double);
  double *host_cube, *device_cube;
  double start, end;
  tcaHandle *cube_handle;
  int desc_tag[2] = {0, 0};
  int dma_slot = 0;
  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_cube, cube_byte));
  TCA_SAFE_CALL(tcaMalloc((void**)&device_cube, cube_byte, tcaMemoryGPU));
  tcaCreateHandleList(&cube_handle, 5, device_cube, cube_byte);
  TCA_SAFE_CALL(tcaCreateDMADesc(&desc_tag[0], 1024));
  TCA_SAFE_CALL(tcaCreateDMADesc(&desc_tag[1], 1024));

  off_t src_offset_hi = 0;
  off_t dst_offset_hi = (n-1)*n*sizeof(double);
  off_t src_offset_lo = (n-1)*n*sizeof(double);
  off_t dst_offset_lo = 0;
  size_t pitch = (n*n)*sizeof(double);
  size_t width = n*sizeof(double);

  if(my_rank == 0 || my_rank == 2 || my_rank == 4){
    TCA_SAFE_CALL(tcaSetDMADesc_Memcpy2D(desc_tag[0], dma_slot, &dma_slot,
    					 &cube_handle[2], dst_offset_lo, pitch,
    					 &cube_handle[my_rank], src_offset_lo, pitch,
    					 width, n, DMA_FLAG, 0, WAIT_TAG));
    TCA_SAFE_CALL(tcaSetDMADesc_Memcpy2D(desc_tag[1], dma_slot, &dma_slot,
					 &cube_handle[4], dst_offset_hi, pitch,
					 &cube_handle[my_rank], src_offset_hi, pitch,
					 width, n, DMA_FLAG, 0, WAIT_TAG));
    TCA_SAFE_CALL(tcaSetDMAChain(DMA_CH, desc_tag[0]));
    TCA_SAFE_CALL(tcaSetDMAChain(DMA_CH, desc_tag[1]));
  }

  for(int z=0; z<n; z++)
    for(int y=0; y<n; y++)
      for(int x=0; x<n; x++)
	host_cube[z*(n*n)+y*n+x] = (double)(my_rank+1);

  CUDA_SAFE_CALL(cudaMemcpy(device_cube, host_cube, cube_byte, cudaMemcpyDefault));
  MPI_Barrier(MPI_COMM_WORLD);

  for(int t=0; t<TIMES+WARMUP; t++){
    if(t == WARMUP){
      MPI_Barrier(MPI_COMM_WORLD);
      start = MPI_Wtime();
    }

    if(my_rank == 0){
      MPI_Request req[2];
      MPI_SAFE_CALL(MPI_Isend(&device_cube[0],         n*n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &req[0]));
      MPI_SAFE_CALL(MPI_Isend(&device_cube[(n-1)*n*n], n*n, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, &req[1]));

      TCA_SAFE_CALL(tcaStartDMADesc(DMA_CH));
      TCA_SAFE_CALL(tcaWaitDMARecvDesc(&cube_handle[my_rank], 0, WAIT_TAG));

      MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    }
    else if(my_rank == 1){
      MPI_SAFE_CALL(MPI_Recv(&device_cube[(n-1)*n*n], n*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    else if(my_rank == 3){
      MPI_SAFE_CALL(MPI_Recv(&device_cube[0],         n*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    else if(my_rank == 2){
      TCA_SAFE_CALL(tcaWaitDMARecvDesc(&cube_handle[0], 0, WAIT_TAG));
    }
    else if(my_rank == 4){
      TCA_SAFE_CALL(tcaWaitDMARecvDesc(&cube_handle[0], 0, WAIT_TAG));
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();
  CUDA_SAFE_CALL(cudaMemcpy(host_cube, device_cube, cube_byte, cudaMemcpyDefault));
  verify(host_cube, n, my_rank);

  double one_way_comm_time = ((end - start)/TIMES/2)*1e6;
  double bandwidth         = matrix_byte / one_way_comm_time;
  if(my_rank == 0 && output_flag == 1)
    printf("N = %d one_way_comm_time = %lf [usec], bandwidth = %lf [MB/s]\n", n, one_way_comm_time, bandwidth);

  TCA_SAFE_CALL(tcaDestroyDMADesc(desc_tag[0]));
  TCA_SAFE_CALL(tcaDestroyDMADesc(desc_tag[1]));
  CUDA_SAFE_CALL(cudaFreeHost(host_cube));
  TCA_SAFE_CALL(tcaFree(device_cube, tcaMemoryGPU));
}

int main(int argc, char** argv)
{
  int my_rank; 
  MPI_SAFE_CALL(MPI_Init(&argc, &argv));
  TCA_SAFE_CALL(tcaInit());
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
  CUDA_SAFE_CALL(cudaSetDevice(0));

  stencil(2, my_rank, 0); // Dry run
  for(int count=2; count<=COUNT; count*=2)
    stencil(count, my_rank, 1);
  
  MPI_SAFE_CALL(MPI_Finalize());
  
  return 0;
}
