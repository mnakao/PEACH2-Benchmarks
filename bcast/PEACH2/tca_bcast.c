#include "common.h"
#define IS_POW_2(n) (!(n & (n-1)))

int main(int argc, char** argv)
{
  int const dmac_ch = 0;
  int const wait_tag = 0x100;
  int my_rank, num_proc, num_procs, namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(&argc, &argv);
  TCA_SAFE_CALL(tcaInit());
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  if(! IS_POW_2(num_proc)){
    printf("This program is executed in only 2^n processes %d\n", num_proc);
    MPI_Finalize();
    exit(1);
  }
  
  MPI_Get_processor_name(processor_name, &namelen);
  printf("Process %d of %d is on %s\n", my_rank, num_proc, processor_name);

  CUDA_SAFE_CALL(cudaSetDevice(0));  
  
  for(int count=1; count<=COUNT; count*=2){
    double start_time, my_time, sum_time;
    size_t byte = count * sizeof(double);
 
    double *host_array, *device_array;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&host_array, byte));
    TCA_SAFE_CALL(tcaMalloc((void**)&device_array, byte, tcaMemoryGPU));

    if(my_rank == 0) for(int i=0; i<count; i++) host_array[i] = (double)i;
    else             for(int i=0; i<count; i++) host_array[i] = -1.0;

    CUDA_SAFE_CALL(cudaMemcpy(device_array, host_array, byte, cudaMemcpyDefault));

    tcaHandle *handle;
    tcaCreateHandleList(&handle, num_proc, device_array, byte);
    tcaDesc *tca_desc = tcaDescNew();
    const int dmaFlag = tcaDMAUseInternal|tcaDMAUseNotifyInternal|tcaDMANotify|tcaDMANotifySelf;

    int steps = (int)log2((double)num_proc);
    int step_size = num_proc;
    for(int i=0;i<steps;i++){
      if(my_rank % step_size == 0){
	int other = my_rank + step_size/2;
	TCA_SAFE_CALL(tcaDescSetMemcpy(tca_desc, &handle[other], 0, &handle[my_rank], 0,
				       byte, dmaFlag, 0, wait_tag));
      }
      step_size /= 2;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(my_rank % 2 == 0)
      TCA_SAFE_CALL(tcaDescSet(tca_desc, dmac_ch));

    sum_time = 0.0;
    for(int t=0; t<TIMES+WARMUP; t++){
      MPI_Barrier(MPI_COMM_WORLD);

      if(t >= WARMUP)
    	start_time = MPI_Wtime();

      int step_size = num_proc;
      for(int i=0;i<steps;i++){
	if(my_rank % step_size == 0){
	  if(my_rank != 0){
	    int other = my_rank - step_size;
	    TCA_SAFE_CALL(tcaWaitDMARecvDesc(&handle[other], 0, wait_tag));
	  }
	  TCA_SAFE_CALL(tcaStartDMADesc(dmac_ch));
	  TCA_SAFE_CALL(tcaWaitDMARecvDesc(&handle[my_rank], 0, wait_tag));
	  break;
	}
	step_size /= 2;
      }
      if(my_rank % 2 == 1){
	int other = my_rank - step_size;
	TCA_SAFE_CALL(tcaWaitDMARecvDesc(&handle[other], 0, wait_tag));
      }

      if(t >= WARMUP){
	my_time = MPI_Wtime() - start_time;
	MPI_Allreduce(MPI_IN_PLACE, &my_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	sum_time += my_time;
      }
    }

    if(my_rank == 0)
      printf("%zd %lf\n", byte, (sum_time/TIMES)*1e6);

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

  MPI_Finalize();

  return 0;
}
