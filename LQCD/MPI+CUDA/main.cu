/* ****************************************************** */
/*    Wilson fermion solver in C language                 */
/*                                                        */
/*    OpenACC benchmark [5 May 0216 H.Matsufuru]          */
/*                                                        */
/*                     Copyright(c) Hideo Matsufuru 2016  */
/* ****************************************************** */

#include "lattice.h"
int left, right, up, down;
real_t *vt;
real_t *tmp_QCDSpinor_s, *tmp_QCDSpinor_r;
real_t *tmp_QCDMatrix_s, *tmp_QCDMatrix_r;
MPI_Request req_u[8], req_mat[2], req_w[4][4], req_spr[4], req_vt[4];
real_t *p, *x;

static real_t u[NDF*NST2*4];
static real_t corr[LT];
#ifdef _PROF
double prof_t[PROF_NUMS];
#endif
MPI_Comm comm_ud, comm_lr;

double dtime()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return ((double)(tv.tv_sec) + (double)(tv.tv_usec) * 1.0e-6);
}

static void create_newcomm(const int pt, const int pz, const int me)
{
  /*
  0 4
  1 5
  2 6
  3 7  PT = 2, PZ = 4 */

  MPI_Comm_split(MPI_COMM_WORLD, me%pz, me/pz, &comm_lr); // color, key
  MPI_Comm_split(MPI_COMM_WORLD, me/pz, me%pz, &comm_ud); // color, key
}

static void uinit(const int me, const int pz, real_t *u)
{
  int Nx2 =  8;
  int Ny2 =  8;
  int Nz2 =  8;
  int Nt2 = 16;
  int Nst2 = Nx2 * Ny2 * Nz2 * Nt2;

  FILE *fp;
  fp = fopen("conf_08080816.txt","r");
  double *ur = (double*)malloc(sizeof(double) * NDF * 4 * Nst2);

  for(int ist = 0; ist < Nst2; ist++){
    for(int idir = 0; idir < 4; idir++){
      for(int idf = 0; idf < NDF; idf++){
	int i = idf + ist*NDF + idir*NDF*Nst2;
	int ret = fscanf(fp, "%lf", &ur[i]);
	if(!ret){
          fprintf(stderr, "Read Error!\n");
          MPI_Finalize();
          exit(0);
	}
      }
    }
  }

  fclose(fp);

  int idir, it, iz, iy, ix;
  for(idir = 0; idir < 4; idir++){
    for(it = 1; it < LT2-1; it++){
      for(iz = 1; iz < LZ2-1; iz++){
	for(iy = 0; iy < NY; iy++){
	  for(ix = 0; ix < NX; ix++){
	    int ist = ix + NX*(iy + NY*(iz + LZ2*it));
	    int ix2 = ix % Nx2;
	    int iy2 = iy % Ny2;
            int iz2 = ((iz-1)+((me%pz)*(LZ2-2))) % Nz2;
            int it2 = ((it-1)+((me/pz)*(LT2-2))) % Nt2;
	    int ist2 = ix2 + Nx2*(iy2 + Ny2*(iz2 + Nz2*it2));
	    for(int idf = 0; idf < NDF; idf++){
	      int i  = idf + NDF*(ist  + idir*NX*NY*LZ2*LT2);
	      int i2 = idf + NDF*(ist2 + idir*Nst2);
	      u[i] = (real_t)ur[i2];
	    }
	  }
	}
      }
    }
  }
  free(ur);
}

__device__ static void setconst(real_t *v, const real_t a)
{
  int i = IDXV(threadIdx.x, blockIdx.x, blockDim.x);
  
  while(i < (LT2-2)*(LZ2-2)*yx_Spinor){
    int t = i / ((LZ2-2)*yx_Spinor);
    int z = (i - t * (LZ2-2)*yx_Spinor)/yx_Spinor;   // (i % ((LZ2-2)*yx_Spinor)) / yx_Spinor;
    int offset = i % yx_Spinor;
    v[(t+1)*LZ2*yx_Spinor + (z+1)*yx_Spinor + offset] = a;
    i += blockDim.x * gridDim.x;
  }
}

__global__ static void set_src(const int me, const int ic, const int id, const int ix, const int iy, const int iz, const int it,
			       real_t *v)
{
  setconst(v, 0.0);
  if(me == 0){ // fix me
    if(threadIdx.x == 0 && blockIdx.x == 0){
      int i = 2*ic + id*NVC + NVC*ND*(ix + iy*NX + (iz+1)*NX*NY + (it+1)*NX*NY*LZ2);
      v[i] = 1.0;
    }
  }
}

static void test_mult(const int me, real_t *u)
{
  int nrepeat = 100;
  real_t *bq2, *xq2;
  HANDLE_ERROR( cudaMalloc( (void**)&bq2, NVST2*sizeof(real_t) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&xq2, NVST2*sizeof(real_t) ) );
  set_src <<< NUM_GANGS, VECTOR_LENGTH >>> (me, 0, 0, 0, 0, 0, 0, bq2);

  int QCDSpinor_zyxvec = LZ * yx_Spinor;
  MPI_Recv_init(bq2 + ((LT2-1)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, &req_w[2][0]);
  MPI_Recv_init(bq2 + yx_Spinor,                   QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, &req_w[2][1]);
  MPI_Send_init(bq2 + (LZ2 + 1)*yx_Spinor,         QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, &req_w[2][2]);
  MPI_Send_init(bq2 + ((LT2-2)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, &req_w[2][3]);

  MPI_Recv_init(xq2 + ((LT2-1)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, &req_w[3][0]);
  MPI_Recv_init(xq2 + yx_Spinor,                   QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, &req_w[3][1]);
  MPI_Send_init(xq2 + (LZ2 + 1)*yx_Spinor,         QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, &req_w[3][2]);
  MPI_Send_init(xq2 + ((LT2-2)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, &req_w[3][3]);

  MPI_Barrier(MPI_COMM_WORLD);
  double time0 = dtime();
  for(int i=0; i<nrepeat; i++){
    opr_DdagD_alt(xq2, u, bq2, 2);
    opr_DdagD_alt(bq2, u, xq2, 3);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double time_tot  = dtime() - time0;
  double fop_mult1 = 2.0 * 1392.0 * (double)(NST);
  double fop_mult  = (double)nrepeat * 2.0 * fop_mult1;

  if(me == 0){
    printf("\nperformance of mult on Host:\n");
    printf("  elapsed time for solver   = %f\n",  time_tot);
    printf("  floating point operations = %f\n",  fop_mult);
    printf("  performance of mult = %f GFlops\n", fop_mult/time_tot * 1.0e-9);
  }

  HANDLE_ERROR( cudaFree(bq2) );
  HANDLE_ERROR( cudaFree(xq2) );
}

int main(int argc, char *argv[])
{
  real_t enorm = 1.E-16;
  real_t diff;
  int nconv;

  int  namelen, me, nprocs;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Get_processor_name(processor_name, &namelen);
  printf("Process %d of %d is on %s\n", me, nprocs, processor_name);

  //  cudaSetDevice(me%NGPUS);

  HANDLE_ERROR( cudaMalloc((void**)&vt, NVST2*sizeof(real_t)) );
  HANDLE_ERROR( cudaMalloc((void**)&tmp_QCDSpinor_s, 2*LT*yx_Spinor * sizeof(real_t)) );
  HANDLE_ERROR( cudaMalloc((void**)&tmp_QCDSpinor_r, 2*LT*yx_Spinor * sizeof(real_t)) );
  HANDLE_ERROR( cudaMalloc((void**)&tmp_QCDMatrix_s, 4*LT*yx_Matrix * sizeof(real_t)) );
  HANDLE_ERROR( cudaMalloc((void**)&tmp_QCDMatrix_r, 4*LT*yx_Matrix * sizeof(real_t)) );

  if(me == 0){
    printf("Simple Wilson solver\n\n");
    printf("NX = %3d, NY = %3d, NZ = %3d, NT = %3d\n", NX, NY, NZ, NT);
    printf("LX = %3d, LY = %3d, LZ = %3d, LT = %3d\n", NX, NY, LZ, LT);
    printf("(PT x PZ) = (%d x %d)\n", PT, PZ);
    printf("CKs = %10.6f\n", CKs);
    printf("enorm = %12.4e\n", enorm);
    printf("NUM=%d LEN=%d\n", NUM_GANGS, VECTOR_LENGTH);
  }

  for(int it = 0; it < LT; it++)
    corr[it] = 0.0;
  
  real_t *u_dev, *xq_dev, *bq_dev;
  HANDLE_ERROR( cudaMalloc( (void**)&u_dev,  4*LT2*LZ2*yx_Matrix*sizeof(real_t) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&xq_dev, NVST2*sizeof(real_t) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&bq_dev, NVST2*sizeof(real_t) ) );
  create_newcomm(PT, PZ, me);
  create_cart(PT, PZ, me);

  int QCDSpinor_zyxvec = LZ * yx_Spinor;
  int QCDMatrix_zyxvec = LZ * yx_Matrix;
  int QCDSpinor_tyxvec = LT * yx_Spinor;
  int QCDMatrix_tyxvec = 4*LT*yx_Matrix;

  HANDLE_ERROR( cudaMalloc((void**)&x, NVST2*sizeof(real_t)) );
  HANDLE_ERROR( cudaMalloc((void**)&p, NVST2*sizeof(real_t)) );

  // Sleeve exchange
  for(int i=0;i<4;i++){
    MPI_Recv_init(u_dev + (i*LT2*LZ2 + 1)*yx_Matrix,               QCDMatrix_zyxvec, MPI_DOUBLE, left,  i, comm_lr, &req_u[i*2]);
    MPI_Send_init(u_dev + (i*LT2*LZ2 + (LT2-2)*LZ2 + 1)*yx_Matrix, QCDMatrix_zyxvec, MPI_DOUBLE, right, i, comm_lr, &req_u[1+i*2]);
  }
  MPI_Recv_init(tmp_QCDMatrix_r, QCDMatrix_tyxvec, MPI_DOUBLE, up,   5, comm_ud, &req_mat[0]);
  MPI_Send_init(tmp_QCDMatrix_s, QCDMatrix_tyxvec, MPI_DOUBLE, down, 5, comm_ud, &req_mat[1]);

  MPI_Recv_init(x + ((LT2-1)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, &req_w[0][0]);
  MPI_Recv_init(x + yx_Spinor,                   QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, &req_w[0][1]);
  MPI_Send_init(x + (LZ2 + 1)*yx_Spinor,         QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, &req_w[0][2]);
  MPI_Send_init(x + ((LT2-2)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, &req_w[0][3]);

  MPI_Recv_init(p + ((LT2-1)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, &req_w[1][0]);
  MPI_Recv_init(p + yx_Spinor,                   QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, &req_w[1][1]);
  MPI_Send_init(p + (LZ2 + 1)*yx_Spinor,         QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, &req_w[1][2]);
  MPI_Send_init(p + ((LT2-2)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, &req_w[1][3]);

  MPI_Recv_init(tmp_QCDSpinor_r + LT*yx_Spinor, QCDSpinor_tyxvec, MPI_DOUBLE, down, 12, comm_ud, &req_spr[0]);
  MPI_Recv_init(tmp_QCDSpinor_r,                QCDSpinor_tyxvec, MPI_DOUBLE, up,   13, comm_ud, &req_spr[1]);
  MPI_Send_init(tmp_QCDSpinor_s,                QCDSpinor_tyxvec, MPI_DOUBLE, up,   12, comm_ud, &req_spr[2]);
  MPI_Send_init(tmp_QCDSpinor_s + LT*yx_Spinor, QCDSpinor_tyxvec, MPI_DOUBLE, down, 13, comm_ud, &req_spr[3]);

  MPI_Recv_init(vt + ((LT2-1)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, &req_vt[0]);
  MPI_Recv_init(vt + yx_Spinor,                   QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, &req_vt[1]);
  MPI_Send_init(vt + (LZ2 + 1)*yx_Spinor,         QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, &req_vt[2]);
  MPI_Send_init(vt + ((LT2-2)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, &req_vt[3]);

  uinit(me, PZ, u);
  HANDLE_ERROR( cudaMemcpy(u_dev, u, 4*LT2*LZ2*yx_Matrix*sizeof(real_t), cudaMemcpyHostToDevice) );
  test_mult(me, u_dev);

  if(me == 0){
    printf("Solver:\n");
    printf("  ic  id   nconv      diff\n");
  }
  double time_tot = 0.0;
  double fop_tot  = 0.0;
  
  for(int ic = 0; ic < NCOL; ic++){
    for(int id = 0; id < ND; id++){
      set_src<<< NUM_GANGS,VECTOR_LENGTH >>>(me, ic, id, 0, 0, 0, 0, bq_dev);
      MPI_Barrier(MPI_COMM_WORLD);
      double time0 = dtime();
      solve_CG(enorm, &nconv, &diff, xq_dev, u_dev, bq_dev);
      MPI_Barrier(MPI_COMM_WORLD);
      double time1 = dtime();
      time_tot += time1 - time0;
      
      if(me == 0)
	printf(" %3d %3d  %6d %12.4e\n", ic, id, nconv, diff);

      double fop_mult1 = 2.0 * 1392.0 * (double)(NST);
      double fop_mult  = (double)(nconv+2) * fop_mult1;
      double fop_lin   = (double)(4+(nconv+1)*11) * (double)(NVST);
      fop_tot  += fop_lin + fop_mult;

      norm2_t(corr, xq_dev);
    }
  }

  real_t corr2[NT];

  if(PZ != 1)
    MPI_Allreduce(MPI_IN_PLACE, corr, LT, MPI_DOUBLE, MPI_SUM, comm_ud);

  if(PT != 1)
    MPI_Allgather(corr, LT, MPI_DOUBLE, corr2, LT, MPI_DOUBLE, comm_lr);
  else
    memcpy(corr2, corr, sizeof(real_t)*LT);

  if(me == 0){  
    printf("\nperformance of solver:\n");
    printf("  elapsed time for solver   = %f\n", time_tot);
    printf("  floating point operations = %f\n", fop_tot);
    printf("  performance of solver = %f GFlops\n", fop_tot/time_tot * 1.0e-9);
    printf("\nsolution squared at each time slice:\n");
    for(int it = 0; it < NT; it++)
      printf(" %6d   %16.8e\n", it, corr2[it]);
  }

#ifdef _PROF
  double prof_t_max[PROF_NUMS], prof_t_min[PROF_NUMS], prof_t_ave[PROF_NUMS];
  MPI_Allreduce(prof_t, prof_t_max, PROF_NUMS, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(prof_t, prof_t_min, PROF_NUMS, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(prof_t, prof_t_ave, PROF_NUMS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  for(int i=0;i<PROF_NUMS;i++)
    prof_t_ave[i] /= nprocs;

  if(me == 0)
  {
    printf("MAX: PACK %f COMM %f OPR %f COPY %f AXPY %f NORM %f DOT %f SCAL %f\n",
           prof_t_max[PACK], prof_t_max[COMM], prof_t_max[OPR], prof_t_max[COPY],
           prof_t_max[AXPY], prof_t_max[NORM], prof_t_max[DOT], prof_t_max[SCAL]);

    printf("MIN: PACK %f COMM %f OPR %f COPY %f AXPY %f NORM %f DOT %f SCAL %f\n",
           prof_t_min[PACK], prof_t_min[COMM], prof_t_min[OPR], prof_t_min[COPY],
           prof_t_min[AXPY], prof_t_min[NORM], prof_t_min[DOT], prof_t_min[SCAL]);

    printf("AVE: PACK %f COMM %f OPR %f COPY %f AXPY %f NORM %f DOT %f SCAL %f\n",
           prof_t_ave[PACK], prof_t_ave[COMM], prof_t_ave[OPR], prof_t_ave[COPY],
           prof_t_ave[AXPY], prof_t_ave[NORM], prof_t_ave[DOT], prof_t_ave[SCAL]);
  }
#endif
  MPI_Finalize();
  return 0;
}
