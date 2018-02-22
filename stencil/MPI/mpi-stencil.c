#include "common.h"
extern void call_pack(double* __restrict__, double* __restrict__, const int, const int, const int);
extern void call_unpack(double* __restrict__, double* __restrict__, const int, const int, const int);

void verify(double *host, int row, int column, int depth, int my_rank)
{
  for(int i=0; i<row; i++) 
    for(int j=0; j<column-1; j++) 
      for(int k=0; k<depth; k++){
	if(fabs(host[i*(column*depth)+j*depth+k] - (double)((my_rank+1)*(i*(column*depth)+j*depth+k))) > 1e-18)
	  printf("Error1\n");
      }

  int target = (my_rank+1)%2;  
  for(int i=0; i<row; i++){
    int j = column-1;
    for(int k=0; k<depth; k++){
      if(fabs(host[i*(column*depth)+j*depth+k] - (double)((target+1)*(i*(column*depth)+0*depth+k))) > 1e-18)
	printf("Error2 [%d] host[%d[%d][%d] %f != %f\n", 
	       my_rank, i, column-1, k, host[i*(column*depth)+j*depth+k], (double)((target+1)*(i*(column*depth)+0*depth+k)));
    }
  }
}

static void stencil(const int count, const int my_rank, const int output_flag)
{
  int row, column, depth;
  row = column = depth = count;
  size_t cube_byte   = row * column * depth * sizeof(double);
  size_t matrix_byte = row * depth * sizeof(double);
  double *host_cube, *device_cube, *tmp_matrix_lo, *tmp_matrix_hi;
  double start, end;
  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_cube, cube_byte));
  CUDA_SAFE_CALL(cudaMalloc((void**)&device_cube,   cube_byte));
  CUDA_SAFE_CALL(cudaMalloc((void**)&tmp_matrix_lo, matrix_byte));
  CUDA_SAFE_CALL(cudaMalloc((void**)&tmp_matrix_hi, matrix_byte));

  for(int i=0; i<row; i++)
    for(int j=0; j<column; j++)
      for(int k=0; k<depth; k++)
	host_cube[i*(column*depth)+j*depth+k] = (double)((my_rank+1)*(i*(column*depth)+j*depth+k));

  CUDA_SAFE_CALL(cudaMemcpy(device_cube, host_cube, cube_byte, cudaMemcpyDefault));
  MPI_Barrier(MPI_COMM_WORLD);

  for(int t=0; t<TIMES+WARMUP; t++){
    if(t == WARMUP){
      MPI_Barrier(MPI_COMM_WORLD);
      start = MPI_Wtime();
    }

    if(my_rank == 0){
      MPI_SAFE_CALL(MPI_Isend(device_cube, row*depth, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD));
      MPI_SAFE_CALL(MPI_Isend(device_cube, row*depth, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD));

      call_pack(device_cube, tmp_matrix_lo, row, column, depth);
      MPI_SAFE_CALL(MPI_Isend(tmp_matrix_lo, row*depth, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD));

      call_pack(device_cube, tmp_matrix_hi, row, column, depth);
      MPI_SAFE_CALL(MPI_Isnd(tmp_matrix_hi, row*depth, MPI_DOUBLE, 4, 0, MPI_COMM_WORLD));

      wait(4)

	
      MPI_SAFE_CALL(MPI_Recv(tmp_matrix, row*depth, MPI_DOUBLE, target, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      call_unpack(tmp_matrix, device_cube, row, column, depth);
    }
    else{
      MPI_SAFE_CALL(MPI_Recv(tmp_matrix, row*depth, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      call_unpack(tmp_matrix, device_cube, row, column, depth);

      call_pack(device_cube, tmp_matrix, row, column, depth);
      MPI_SAFE_CALL(MPI_Send(tmp_matrix, row*depth, MPI_DOUBLE, target, 1, MPI_COMM_WORLD));
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();
  CUDA_SAFE_CALL(cudaMemcpy(host_cube, device_cube, cube_byte, cudaMemcpyDefault));
  verify(host_cube, row, column, depth, my_rank);

  double one_way_comm_time = ((end - start)/TIMES/2)*1e6;
  double bandwidth         = matrix_byte / one_way_comm_time;
  if(my_rank == 0 && output_flag == 1)
    printf("N = %d one_way_comm_time = %lf [usec], bandwidth = %lf [MB/s]\n", count, one_way_comm_time, bandwidth);

  CUDA_SAFE_CALL(cudaFreeHost(host_cube));
  CUDA_SAFE_CALL(cudaFree(device_cube));
  CUDA_SAFE_CALL(cudaFree(tmp_matrix));
}

int main(int argc, char** argv)
{
  int my_rank; 
  MPI_SAFE_CALL(MPI_Init(&argc, &argv));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
  CUDA_SAFE_CALL(cudaSetDevice(0));

  stencil(2, my_rank, 0); // Dry run
  for(int count=2; count<=COUNT; count*=2)
    stencil(count, my_rank, 1);
  
  MPI_SAFE_CALL(MPI_Finalize());
  
  return 0;
}
