#include "common.h"
#define IS_POW_2(n) (!(n & (n-1)))
#define WAIT_TAG (0x100)
#define DMA_FLAG (tcaDMAUseInternal|tcaDMAUseNotifyInternal|tcaDMANotify|tcaDMANotifySelf)
void tca_allgather(const int count, const int my_rank, const int num_procs, const int output_flag);

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

  tca_allgather(2, my_rank, num_procs, 0); // The first run is slow, so run it empty.
  for(int count=2; count<=COUNT; count*=2)
    tca_allgather(count, my_rank, num_procs, 1);
  
  MPI_Finalize();

  return 0;
}

static void tca_allgather(const int count, const int my_rank, const int num_procs, const int output_flag)
{
  double start_time, my_time, sum_time;
  size_t send_byte  = count * sizeof(double);
  size_t whole_byte = count * sizeof(double) * num_procs;

  double *host_send_array,  *device_send_array;
  double *host_whole_array, *device_whole_array;
  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_send_array,  send_byte));
  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_whole_array, whole_byte));
  TCA_SAFE_CALL(tcaMalloc((void**)&device_send_array,  send_byte,  tcaMemoryGPU));
  TCA_SAFE_CALL(tcaMalloc((void**)&device_whole_array, whole_byte, tcaMemoryGPU));

  for(int i=0; i<count; i++)
    host_send_array[i] = (double)my_rank;

  CUDA_SAFE_CALL(cudaMemcpy(device_send_array, host_send_array, send_byte, cudaMemcpyDefault));

  tcaHandle send_handle, *whole_handle;
  TCA_SAFE_CALL(tcaCreateHandle(&send_handle, device_send_array, send_byte, tcaMemoryGPU));
  tcaCreateHandleList(&whole_handle, num_procs, device_whole_array, whole_byte);

  int steps = (int)log2((double)num_procs);
  tcaDesc *tca_desc[steps];
  for(int i=0;i<steps;i++)
    tca_desc[i] = tcaDescNew();

  off_t my_offset   = my_rank * send_byte;
  TCA_SAFE_CALL(tcaDescSetMemcpy(tca_desc[0], &whole_handle[my_rank], my_offset, &send_handle, 0,
				 send_byte, DMA_FLAG, 0, WAIT_TAG));
  TCA_SAFE_CALL(tcaDescSet(tca_desc[0], 0));
  
  int rank_group = my_rank;
  for(int i=0, mask=1; mask<num_procs; mask<<=1, i++){
    int other    = my_rank ^ mask;
    off_t offset = rank_group * send_byte;
    TCA_SAFE_CALL(tcaDescSetMemcpy(tca_desc[i], &whole_handle[other], offset,
				   &whole_handle[my_rank], offset, (mask*send_byte),
				   DMA_FLAG, 0, WAIT_TAG));
    TCA_SAFE_CALL(tcaDescSet(tca_desc[i], i));
    if(my_rank >= other)
      rank_group -= mask;
  }
  
  sum_time = 0.0;
  for(int t=0; t<TIMES+WARMUP; t++){
    MPI_Barrier(MPI_COMM_WORLD);
    if(t >= WARMUP)
      start_time = MPI_Wtime();
    
    for(int i=0, mask=1; mask<num_procs; mask<<=1, i++) {
      int other = my_rank ^ mask;
      TCA_SAFE_CALL(tcaStartDMADesc(i));
      TCA_SAFE_CALL(tcaWaitDMARecvDesc(&whole_handle[my_rank], 0, WAIT_TAG));
      TCA_SAFE_CALL(tcaWaitDMARecvDesc(&whole_handle[other],   0, WAIT_TAG));
    }
    
    if(t >= WARMUP){
      my_time = MPI_Wtime() - start_time;
      MPI_Allreduce(MPI_IN_PLACE, &my_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      sum_time += my_time;
    }
  }
  
  if(my_rank == 0 && output_flag == 1)
    printf("%zd %lf us\n", send_byte, (sum_time/TIMES)*1e6);
  
  CUDA_SAFE_CALL(cudaMemcpy(host_whole_array, device_whole_array, whole_byte, cudaMemcpyDefault));

  for(int i=0; i<num_procs; i++)
    for(int j=0; j<count; j++)
      if(fabs(host_whole_array[i*count+j] - (double)i) > 1e-16){
	printf("Incorrect answer.\n");
	break;
      }
  
  MPI_Barrier(MPI_COMM_WORLD);
  free(whole_handle);
  CUDA_SAFE_CALL(cudaFreeHost(host_send_array));
  CUDA_SAFE_CALL(cudaFreeHost(host_whole_array));
  TCA_SAFE_CALL(tcaFree(device_send_array,  tcaMemoryGPU));
  TCA_SAFE_CALL(tcaFree(device_whole_array, tcaMemoryGPU));
  for(int i=0;i<steps;i++)
    TCA_SAFE_CALL(tcaDescFree(tca_desc[i]));
}

