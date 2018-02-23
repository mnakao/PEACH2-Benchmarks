#include "common.h"
extern void call_pack(double* __restrict__, double* __restrict__, const int,  const int);
extern void call_unpack(double* __restrict__, double* __restrict__, const int, const int);

void verify(double *host, int n, int my_rank)
{
  if(my_rank == 1){
    for(int z=0; z<n-1; z++)
      for(int y=0; y<n; y++)
        for(int x=0; x<n; x++)
          if(fabs(host[z*(n*n)+y*n+x] - (double)(my_rank+1)) > 1e-16){
            printf("Error1\n");
            break;
          }
    for(int y=0; y<n; y++)
      for(int x=0; x<n; x++)
	if(fabs(host[(n-1)*(n*n)+y*n+x] - 1.0) > 1e-16){
	  printf("Error2\n");
	  break;
	}
  }
  else if(my_rank == 3){
    for(int z=1; z<n; z++)
      for(int y=0; y<n; y++)
        for(int x=0; x<n; x++)
          if(fabs(host[z*(n*n)+y*n+x] - (double)(my_rank+1)) > 1e-16){
            printf("[%d] Error1\n", my_rank);
            break;
          }
    for(int y=0; y<n; y++)
      for(int x=0; x<n; x++)
        if(fabs(host[y*n+x] - 1.0)) > 1e-16){
	  printf("[%d] Error2\n", my_rank);
          break;
        }
  }
  else if(my_rank == 2){
    for(int z=0; z<n; z++)
      for(int y=1; y<n; y++)
        for(int x=0; x<n; x++)
          if(fabs(host[z*(n*n)+y*n+x] - (double)(my_rank+1)) > 1e-16){
	    printf("[%d] Error1\n", my_rank);
            break;
          }
    for(int z=0; z<n; z++)
      for(int x=0; x<n; x++)
        if(fabs(host[z*(n*n)+x] - 1.0) > 1e-16){
	    printf("[%d] Error2\n", my_rank);
            break;
          }
  }
  else if(my_rank == 4){
    for(int z=0; z<n; z++)
      for(int y=0; y<n-1; y++)
        for(int x=0; x<n; x++)
          if(fabs(host[z*(n*n)+y*n+x] - (double)(my_rank+1)) > 1e-16){
	    printf("[%d] Error1\n", my_rank);
            break;
          }
    for(int z=0; z<n; z++)
      for(int x=0; x<n; z++)
        if(fabs(host[z*(n*n)+(n-1)*n+x] - 1.0) > 1e-16){
	  printf("[%d] Error2\n", my_rank);
	  break;
	}
  }
}

static void stencil(const int n, const int my_rank, const int output_flag)
{
  size_t cube_byte   = (n * n * n) * sizeof(double);
  size_t matrix_byte = (n * n) * sizeof(double);
  double *host_cube, *device_cube, *tmp_matrix_lo, *tmp_matrix_hi;
  double start, end;
  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_cube, cube_byte));
  CUDA_SAFE_CALL(cudaMalloc((void**)&device_cube,   cube_byte));
  CUDA_SAFE_CALL(cudaMalloc((void**)&tmp_matrix_lo, matrix_byte));
  CUDA_SAFE_CALL(cudaMalloc((void**)&tmp_matrix_hi, matrix_byte));

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
      MPI_Request req[4];
      MPI_SAFE_CALL(MPI_Isend(&device_cube[0],         n*n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &req[0]));
      MPI_SAFE_CALL(MPI_Isend(&device_cube[(n-1)*n*n], n*n, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, &req[1]));

      call_pack(device_cube, tmp_matrix_lo, n, n-1);
      MPI_SAFE_CALL(MPI_Isend(tmp_matrix_hi,           n*n, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, &req[2]));

      call_pack(device_cube, tmp_matrix_lo, n, 0);
      MPI_SAFE_CALL(MPI_Isend(tmp_matrix_lo,           n*n, MPI_DOUBLE, 4, 0, MPI_COMM_WORLD, &req[3]));

      MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    }
    else if(my_rank == 1){
      MPI_SAFE_CALL(MPI_Recv(&device_cube[(n-1)*n*n], n*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    else if(my_rank == 3){
      MPI_SAFE_CALL(MPI_Recv(&device_cube[0],         n*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    else if(my_rank == 2){
      MPI_SAFE_CALL(MPI_Recv(tmp_matrix_lo,           n*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      call_unpack(tmp_matrix_lo, device_cube, n, 0);
    }
    else if(my_rank == 4){
      MPI_SAFE_CALL(MPI_Recv(tmp_matrix_hi,           n*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      call_unpack(tmp_matrix_hi, device_cube, n, n-1);
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

  CUDA_SAFE_CALL(cudaFreeHost(host_cube));
  CUDA_SAFE_CALL(cudaFree(device_cube));
  CUDA_SAFE_CALL(cudaFree(tmp_matrix_lo));
  CUDA_SAFE_CALL(cudaFree(tmp_matrix_hi));
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
