#include "common.h"
#define IS_POW_2(n) (!(n & (n-1)))
#define WAIT_TAG (0x100)
#define DMA_FLAG (tcaDMAUseInternal|tcaDMAUseNotifyInternal|tcaDMANotify|tcaDMANotifySelf)
#define DMAC_CH (0)
void tca_bcast(const int count, const int my_rank, const int num_procs, const int output_flag);

int main(int argc, char** argv)
{
  int my_rank, num_procs, namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(&argc, &argv);
  TCA_SAFE_CALL(tcaInit());
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  if(! IS_POW_2(num_procs)){
    if(my_rank == 0)
      printf("This program is executed in only 2^n processes %d\n", num_procs);
    MPI_Finalize();
    exit(1);
  }
  
  MPI_Get_processor_name(processor_name, &namelen);
  printf("Process %d of %d is on %s\n", my_rank, num_procs, processor_name);

  CUDA_SAFE_CALL(cudaSetDevice(0));
  
  tca_bcast(1, my_rank, num_procs, 0);
  for(int count=1; count<=COUNT; count*=2)
    tca_bcast(count, my_rank, num_procs, 1);

  MPI_Finalize();

  return 0;
}

static void tca_bcast(const int count, const int my_rank, const int num_procs, const int output_flag)
{
  double start_time, my_time, sum_time;
  double *host_array, *device_array;

  size_t byte = count * sizeof(double);
  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_array, byte));
  TCA_SAFE_CALL(tcaMalloc((void**)&device_array, byte, tcaMemoryGPU));

  if(my_rank == 0) for(int i=0; i<count; i++) host_array[i] = (double)i;
  else             for(int i=0; i<count; i++) host_array[i] = -1.0;

  CUDA_SAFE_CALL(cudaMemcpy(device_array, host_array, byte, cudaMemcpyDefault));

  tcaHandle *handle;
  tcaCreateHandleList(&handle, num_procs, device_array, byte);
  tcaDesc *tca_desc = tcaDescNew();

  int steps = (int)log2((double)num_procs);
  int step_size = num_procs;
  for(int i=0;i<steps;i++){
    if(my_rank % step_size == 0){
      int other = my_rank + step_size/2;
      TCA_SAFE_CALL(tcaDescSetMemcpy(tca_desc, &handle[other], 0, &handle[my_rank], 0,
				     byte, DMA_FLAG, 0, WAIT_TAG));
    }
    step_size /= 2;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(my_rank % 2 == 0)
    TCA_SAFE_CALL(tcaDescSet(tca_desc, DMAC_CH));

  sum_time = 0.0;
  for(int t=0; t<TIMES+WARMUP; t++){
    MPI_Barrier(MPI_COMM_WORLD);

    if(t >= WARMUP)
      start_time = MPI_Wtime();

    int step_size = num_procs;
    for(int i=0;i<steps;i++){
      if(my_rank % step_size == 0){
	if(my_rank != 0){
	  int other = my_rank - step_size;
	  TCA_SAFE_CALL(tcaWaitDMARecvDesc(&handle[other], 0, WAIT_TAG));
	}
	TCA_SAFE_CALL(tcaStartDMADesc(DMAC_CH));
	TCA_SAFE_CALL(tcaWaitDMARecvDesc(&handle[my_rank], 0, WAIT_TAG));
	break;
      }
      step_size /= 2;
    } if(my_rank % 2 == 1){
      int other = my_rank - step_size;
      TCA_SAFE_CALL(tcaWaitDMARecvDesc(&handle[other], 0, WAIT_TAG));
    }
    
    if(t >= WARMUP){
      my_time = MPI_Wtime() - start_time;
      MPI_Allreduce(MPI_IN_PLACE, &my_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      sum_time += my_time;
    }
  }
  
  if(my_rank == 0 && output_flag == 1)
    printf("%zd %lf us\n", byte, (sum_time/TIMES)*1e6);

  CUDA_SAFE_CALL(cudaMemcpy(host_array, device_array, byte, cudaMemcpyDefault));
  for(int i=0; i<count; i++)
    if (fabs(host_array[i] - (double)i) > 1e-18)
      printf("Incorrect answer.\n");

  MPI_Barrier(MPI_COMM_WORLD);

  free(handle);
  CUDA_SAFE_CALL(cudaFreeHost(host_array));
  TCA_SAFE_CALL(tcaFree(device_array, tcaMemoryGPU));
  TCA_SAFE_CALL(tcaDescFree(tca_desc));
}
