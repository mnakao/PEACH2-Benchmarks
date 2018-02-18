#include "common.h"

void verify(double *host, int row, int column, int depth, int my_rank)
{
  for(int i=0; i<row; i++) 
    for(int j=0; j<column-1; j++) 
      for(int k=0; k<depth; k++){
	if(fabs(host[i*(column*depth)+j*depth+k] - (double)((my_rank+1)*(i*(column*depth)+j*depth+k))) > 1e-18)
	  printf("Error\n");
      }

  int other = (my_rank+1)%2;  
  for(int i=0; i<row; i++){
    int j = column-1;
    for(int k=0; k<depth; k++){
      if(fabs(host[i*(column*depth)+j*depth+k] - (double)((other+1)*(i*(column*depth)+0*depth+k))) > 1e-18)
	printf("Error\n");
    }
  }
}

int main(int argc, char** argv)
{
  int row, column, depth, my_rank, other;
  size_t cube_byte, matrix_byte;
  double *host_cube, *device_cube;
  double start, end;
  MPI_Datatype STRIDED_TYPE;
  
  MPI_SAFE_CALL(MPI_Init(&argc, &argv));
  TCA_SAFE_CALL(tcaInit());
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
  other = (my_rank+1)%2;
  CUDA_SAFE_CALL(cudaSetDevice(0));

  for(int count=2; count<=COUNT; count*=2) {
    row         = column = depth = count;
    cube_byte   = row * column * depth * sizeof(double);
    matrix_byte = row * column * sizeof(double);

    CUDA_SAFE_CALL(cudaMallocHost((void**)&host_cube, cube_byte));
    TCA_SAFE_CALL(tcaMalloc((void**)&device_cube, cube_byte, tcaMemoryGPU));

    for(int i=0; i<row; i++)
      for(int j=0; j<column; j++)
	for(int k=0; k<depth; k++)
	  host_cube[i*(column*depth)+j*depth+k] = (double)((my_rank+1)*(i*(column*depth)+j*depth+k));

    CUDA_SAFE_CALL(cudaMemcpy(device_cube, host_cube, cube_byte, cudaMemcpyDefault));

    MPI_SAFE_CALL(MPI_Type_vector(row, depth, (column*depth), MPI_DOUBLE, &STRIDED_TYPE));
    MPI_SAFE_CALL(MPI_Type_commit(&STRIDED_TYPE));
    MPI_Barrier(MPI_COMM_WORLD);

    for(int t=0; t<TIMES+WARMUP; t++){
      if(t == WARMUP){
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
      }

      int src_index = 0;
      int dst_index = (column-1)*depth;
      if(my_rank == 0){
	MPI_SAFE_CALL(MPI_Send(&device_cube[src_index], 1, STRIDED_TYPE, other, 0, MPI_COMM_WORLD));
	MPI_SAFE_CALL(MPI_Recv(&device_cube[dst_index], 1, STRIDED_TYPE, other, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      } 
      else{
	MPI_SAFE_CALL(MPI_Recv(&device_cube[dst_index], 1, STRIDED_TYPE, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
	MPI_SAFE_CALL(MPI_Send(&device_cube[src_index], 1, STRIDED_TYPE, other, 1, MPI_COMM_WORLD));
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
    
    MPI_SAFE_CALL(MPI_Type_free(&STRIDED_TYPE));
    CUDA_SAFE_CALL(cudaFreeHost(host_cube));
    TCA_SAFE_CALL(tcaFree(device_cube, tcaMemoryGPU));
  }
  
  MPI_SAFE_CALL(MPI_Finalize());
  
  return 0;
}
