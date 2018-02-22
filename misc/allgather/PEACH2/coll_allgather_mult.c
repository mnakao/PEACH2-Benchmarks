#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "tca-help.h"
#include "cuda-help.h"

#define COUNT 1024*64
#define WARMUP 10
#define TIMES 100

int main (int argc, char **argv)
{
  int my_rank, num_proc, num_comm;
  int i, t, count;
  static size_t whole_byte, byte;
  static double start, end, my_time, max_time, sum_time;
  double *host_send_buffer, *host_recv_buffer, *host_ans_recv_buffer;
  double *device_send_buffer, *device_recv_buffer, *device_ans_recv_buffer;

  tcaHandle send_handle, *handles;
  int mask, target_rank;
  tcaDesc *tca_desc_0;
  tcaDesc *tca_desc_1;
  tcaDesc *tca_desc_2;
  tcaDesc *tca_desc_3;
  tcaDesc *tca_desc_4;
  int dmac_ch;
  int const wait_tag = 0x100;
  MPI_Request req[2];

  MPI_Init(&argc, &argv);
  TCA_SAFE_CALL(tcaInit());
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  num_comm = (int)log2((double)num_proc);

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
    TCA_SAFE_CALL(tcaCreateHandle(&send_handle, device_send_buffer, byte, tcaMemoryGPU));

    tcaCreateHandleList(&handles, num_proc, device_recv_buffer, whole_byte);

    //    int dmaFlag = tcaDMAUseInternal|tcaDMAUseNotifyInternal|tcaDMANotify;
    const int dmaFlag = tcaDMAUseInternal|tcaDMAUseNotifyInternal|tcaDMANotify|tcaDMAPipeline|tcaDMANotifySelf;
    int rank_group = my_rank;

    tca_desc_0 = tcaDescNew();
    TCA_SAFE_CALL(tcaDescSetMemcpy(tca_desc_0, &handles[my_rank], (rank_group*byte),
				   &send_handle, 0, byte, dmaFlag, 0, wait_tag));

    for (i = 1, mask = 1; mask < num_proc; mask <<= 1, i++) { // mask : 1 -> 2 -> 4 -> 8 -> 16 -> ...
      target_rank = my_rank ^ mask;
      
      off_t offset = rank_group * byte;

      if (i <= num_comm) { // TCA
	if (i == 1) {
	  tca_desc_1 = tcaDescNew();
	  TCA_SAFE_CALL(tcaDescSetMemcpy(tca_desc_1,
					 &handles[target_rank], offset,
					 &handles[my_rank], offset,
					 (mask * byte), dmaFlag, i, wait_tag));
	} else if (i == 2) {
	  tca_desc_2 = tcaDescNew();
	  TCA_SAFE_CALL(tcaDescSetMemcpy(tca_desc_2,
					 &handles[target_rank], offset,
					 &handles[my_rank], offset,
					 (mask * byte), dmaFlag, i, wait_tag));
	} else if (i == 3) {
	  tca_desc_3 = tcaDescNew();
	  TCA_SAFE_CALL(tcaDescSetMemcpy(tca_desc_3,
					 &handles[target_rank], offset,
					 &handles[my_rank], offset,
					 (mask * byte), dmaFlag, i, wait_tag));
	} else if (i == 4) {
	  tca_desc_4 = tcaDescNew();
	  TCA_SAFE_CALL(tcaDescSetMemcpy(tca_desc_4,
					 &handles[target_rank], offset,
					 &handles[my_rank], offset,
					 (mask * byte), dmaFlag, i, wait_tag));
	}
      } else { // MPI
	/* int index = offset / sizeof(double); */
	/* int target_index; */
	/* MPI_Sendrecv(&index, 1, MPI_INT, target_rank, 0, &target_index, 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); */
	/* MPI_Send_init(&device_recv_buffer[index], (mask * byte), MPI_BYTE, target_rank, i, MPI_COMM_WORLD, &req[0]); */
	/* MPI_Recv_init(&device_recv_buffer[target_index], (mask * byte), MPI_BYTE, target_rank, i, MPI_COMM_WORLD, &req[1]); */
      }

      if (my_rank >= target_rank) {
        rank_group -= mask;
      }
    }

    sum_time = 0.0;
    for (t = 0; t < (WARMUP + TIMES); t++) {
      MPI_Barrier(MPI_COMM_WORLD);

      if (t >= WARMUP)
    	start = MPI_Wtime();

      dmac_ch = (num_comm - 1);
      
      TCA_SAFE_CALL(tcaDescSet(tca_desc_0, dmac_ch));
      TCA_SAFE_CALL(tcaStartDMADesc(dmac_ch));
      TCA_SAFE_CALL(tcaWaitDMARecvDesc(&send_handle, 0, wait_tag));

      for (i = 1, mask = 1; mask < num_proc; mask <<= 1, i++) { // mask : 1 -> 2 -> 4 -> 8 -> 16 -> ...
	if (i > 1)
	  dmac_ch--;
      	target_rank = my_rank ^ mask;

	if ( i < num_comm) { // TCA
	  if (i == 1) {
	    TCA_SAFE_CALL(tcaWaitDMAC(dmac_ch));
	    TCA_SAFE_CALL(tcaDescSet(tca_desc_1, dmac_ch));
	  } else if (i == 2) {
	    TCA_SAFE_CALL(tcaDescSet(tca_desc_2, dmac_ch));
	  } else if (i == 3) {
	    TCA_SAFE_CALL(tcaDescSet(tca_desc_3, dmac_ch));
	  } else if (i == 4) {
	    TCA_SAFE_CALL(tcaDescSet(tca_desc_4, dmac_ch));
	  }
	  TCA_SAFE_CALL(tcaStartDMADesc(dmac_ch));
	  TCA_SAFE_CALL(tcaWaitDMARecvDesc(&handles[target_rank], i, wait_tag));
	} else { // MPI
	  /* int error = -99; */
	  /* error = MPI_Startall(2, req); */
	  /* if (error != MPI_SUCCESS) */
	  /*   printf("error = %d\n", error); */

	  /* error = MPI_Waitall(2, req, MPI_STATUS_IGNORE); */
	  /* if (error != MPI_SUCCESS) */
	  /*   printf("error = %d\n", error); */
	  /* MPI_Barrier(MPI_COMM_WORLD); */
	  /* if (my_rank < (num_proc / 2)) { */
	  /*   MPI_Start(&req[0]); */
	  /*   MPI_Wait(&req[0], MPI_STATUS_IGNORE); */
	  /*   MPI_Start(&req[1]); */
	  /*   MPI_Wait(&req[1], MPI_STATUS_IGNORE); */
	  /* } else { */
	  /*   MPI_Start(&req[1]); */
	  /*   MPI_Wait(&req[1], MPI_STATUS_IGNORE); */
	  /*   MPI_Start(&req[0]); */
	  /*   MPI_Wait(&req[0], MPI_STATUS_IGNORE); */
	  /* } */
	}
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
    CUDA_SAFE_CALL(cudaMemcpy(host_ans_recv_buffer, device_ans_recv_buffer, whole_byte, cudaMemcpyDefault));

    /* for (i = 0; i < (count * num_proc); i++) { */
    /*   if (fabs(host_recv_buffer[i] - host_ans_recv_buffer[i]) > 1e16) { */
    /* 	printf("failed\n"); */
    /* 	exit(-1); */
    /*   } */
    /* } */

    if (my_rank == 0) {
      printf("%lf\n", (sum_time / TIMES)*1e6);
      /* for (i = 0; i < (count * num_proc); i++) { */
      /* 	printf("%lf\t", host_recv_buffer[i]); */
      /* } */
      /* printf("\n"); */
    }

    free(handles);

    CUDA_SAFE_CALL(cudaFreeHost(host_send_buffer));
    CUDA_SAFE_CALL(cudaFreeHost(host_recv_buffer));
    CUDA_SAFE_CALL(cudaFreeHost(host_ans_recv_buffer));
    TCA_SAFE_CALL(tcaFree(device_send_buffer, tcaMemoryGPU));
    TCA_SAFE_CALL(tcaFree(device_recv_buffer, tcaMemoryGPU));
    TCA_SAFE_CALL(tcaFree(device_ans_recv_buffer, tcaMemoryGPU));

    TCA_SAFE_CALL(tcaDescFree(tca_desc_0));
    TCA_SAFE_CALL(tcaDescFree(tca_desc_1));
    if (num_proc == 8 || num_proc == 16) {
      TCA_SAFE_CALL(tcaDescFree(tca_desc_2));
    }
    if (num_proc == 16) {
      TCA_SAFE_CALL(tcaDescFree(tca_desc_3));
    }
  }

  
  MPI_Finalize();

  return 0;
}
