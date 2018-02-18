#include "common.h"
#define LAST_FLAG (tcaDMAUseInternal|tcaDMAUseNotifyInternal|tcaDMANotify|tcaDMAPipeline|tcaDMANotifySelf)

void verify(double *host, int row, int column, int depth, int my_rank)
{
  for(int i=0; i<row; i++)
    for(int j=0; j<column-1; j++)
      for(int k=0; k<depth; k++){
        if(fabs(host[i*(column*depth)+j*depth+k] - (double)((my_rank+1)*(i*(column*depth)+j*depth+k))) > 1e-18)
          printf("Error1\n");
      }

  int other = (my_rank+1)%2;
  for(int i=0; i<row; i++){
    int j = column-1;
    for(int k=0; k<depth; k++){
      if(fabs(host[i*(column*depth)+j*depth+k] - (double)((other+1)*(i*(column*depth)+0*depth+k))) > 1e-18)
        printf("Error2 [%d] host[%d[%d][%d] %f != %f\n",
               my_rank, i, column-1, k, host[i*(column*depth)+j*depth+k], (double)((other+1)*(i*(column*depth)+0*depth+k)));
    }
  }
}

int main(int argc, char** argv)
{
  int row, column, depth, my_rank, other;
  size_t cube_byte, matrix_byte;
  double *host_cube, *device_cube;
  double start, end;
  tcaHandle *cube_handle;

  int dma_slot;
  int desc_tag;
  int const dma_ch = 0;
  int const wait_tag = 0x100;

  MPI_SAFE_CALL(MPI_Init(&argc, &argv));
  TCA_SAFE_CALL(tcaInit());
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
  other = (my_rank+1)%2;

  CUDA_SAFE_CALL(cudaSetDevice(0));

  for(int count=2; count<=COUNT; count*= 2){
    row         = column = depth = count;
    cube_byte   = row * column * depth * sizeof(double);
    matrix_byte = row * depth * sizeof(double);

    CUDA_SAFE_CALL(cudaMallocHost((void**)&host_cube, cube_byte));
    TCA_SAFE_CALL(tcaMalloc((void**)&device_cube, cube_byte, tcaMemoryGPU));
    tcaCreateHandleList(&cube_handle, 2, device_cube, cube_byte);
    
    for(int i=0; i<row; i++) 
      for(int j=0; j<column; j++) 
	for(int k=0; k<depth; k++) 
	  host_cube[i*(column*depth)+j*depth+k] = (double)((my_rank+1)*(i*(column*depth)+j*depth+k));

    CUDA_SAFE_CALL(cudaMemcpy(device_cube, host_cube, cube_byte, cudaMemcpyDefault));

    desc_tag = 0;
    TCA_SAFE_CALL(tcaCreateDMADesc(&desc_tag, 1024));

    off_t src_offset = 0;
    off_t dst_offset = (column-1)*depth*sizeof(double);
    size_t pitch = (column*depth)*sizeof(double);
    size_t width = depth*sizeof(double);

    dma_slot = 0;
    TCA_SAFE_CALL(tcaSetDMADesc_Memcpy2D(desc_tag, dma_slot, &dma_slot,
					 &cube_handle[other], dst_offset, pitch,
					 &cube_handle[my_rank], src_offset, pitch,
					 width, row, LAST_FLAG, 0, wait_tag));
    TCA_SAFE_CALL(tcaSetDMAChain(dma_ch, desc_tag));

    MPI_Barrier(MPI_COMM_WORLD);
    for(int t = 0; t <TIMES+WARMUP; t++) {
      if(t == WARMUP){
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
      }

      if(my_rank == 0){
	TCA_SAFE_CALL(tcaStartDMADesc(dma_ch));
	TCA_SAFE_CALL(tcaWaitDMARecvDesc(&cube_handle[my_rank], 0, wait_tag));
	TCA_SAFE_CALL(tcaWaitDMARecvDesc(&cube_handle[other], 0, wait_tag));
      }
      else {
	TCA_SAFE_CALL(tcaWaitDMARecvDesc(&cube_handle[other], 0, wait_tag));
	TCA_SAFE_CALL(tcaStartDMADesc(dma_ch));
	TCA_SAFE_CALL(tcaWaitDMARecvDesc(&cube_handle[my_rank], 0, wait_tag));
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    CUDA_SAFE_CALL(cudaMemcpy(host_cube, device_cube, cube_byte, cudaMemcpyDefault));
    verify(host_cube, row, column, depth, my_rank);

    double one_way_comm_time = ((end - start)/TIMES/2)*1e6;
    double bandwidth         = matrix_byte / one_way_comm_time;
    if(my_rank == 0)
      printf("N = %d one_way_comm_time = %lf [usec], bandwidth = %lf [MB/s]\n", count, one_way_comm_time, bandwidth);

    TCA_SAFE_CALL(tcaDestroyDMADesc(desc_tag));
    CUDA_SAFE_CALL(cudaFreeHost(host_cube));
    TCA_SAFE_CALL(tcaFree(device_cube, tcaMemoryGPU));
  }

  MPI_SAFE_CALL(MPI_Finalize());

  return 0;
}
