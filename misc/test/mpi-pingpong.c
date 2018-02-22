#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define COUNT 1024*512
#define WARMUP 10
#define TIMES 500

static void pingpong(const int count, const int my_rank, const int output_flag)
{
  int target = (my_rank + 1) % 2;
  int transfer_byte = count * sizeof(double);
  double start, end;
  double host_buf[count], dummy_buf[count];

  for(int t=0; t<TIMES+WARMUP; t++) {
    if(t == WARMUP){
      MPI_Barrier(MPI_COMM_WORLD);
      start = MPI_Wtime();
    }
    if(my_rank == 0){
      MPI_Send(host_buf, count, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
      MPI_Recv(dummy_buf, count, MPI_DOUBLE, target, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else{
      MPI_Recv(dummy_buf, count, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(host_buf, count, MPI_DOUBLE, target, 1, MPI_COMM_WORLD);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();

  double one_way_comm_time = ((end-start)/TIMES/2)*1e6;
  double bandwidth = transfer_byte / one_way_comm_time;
  if(my_rank == 0 && output_flag)
    printf("size = %d one_way_comm_time = %lf [usec], bandwidth = %lf [MB/s]\n", transfer_byte, one_way_comm_time, bandwidth);
  }

int main(int argc, char** argv)
{
  int my_rank, namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Get_processor_name(processor_name, &namelen);
  fprintf(stdout,"Process %d on %s\n", my_rank, processor_name);

  for(int count=1; count<=COUNT; count*=2)
    pingpong(count, my_rank, 1);

  MPI_Finalize();

  return 0;
}
