/* ****************************************************** */
/*    Wilson fermion solver in C language                 */
/*                                                        */
/*    OpenACC benchmark [5 May 0216 H.Matsufuru]          */
/*                                                        */
/*                     Copyright(c) Hideo Matsufuru 2016  */
/* ****************************************************** */
#include "lattice.h"

static QCDMatrix_t u[4][LT2][LZ2][NY][NX];
static QCDSpinor_t xq[LT2][LZ2][NY][NX], bq[LT2][LZ2][NY][NX];
static real_t corr[LT];
#ifdef _PROF
double prof_t[PROF_NUMS];
#endif
MPI_Comm comm_ud, comm_lr;

int left, right, up, down;
QCDSpinor_t vt[LT2][LZ2][NY][NX];
QCDSpinor_t tmp_QCDSpinor_s[2][LT][NY][NX], tmp_QCDSpinor_r[2][LT][NY][NX];
QCDMatrix_t tmp_QCDMatrix_s[4][LT][NY][NX], tmp_QCDMatrix_r[4][LT][NY][NX];
QCDSpinor_t p[LT2][LZ2][NY][NX], x[LT2][LZ2][NY][NX];
#pragma acc declare create(vt, tmp_QCDSpinor_s, tmp_QCDSpinor_r, tmp_QCDMatrix_s, tmp_QCDMatrix_r, p, x)
MPI_Request req_u[8], req_mat[2], req_w[4][4], req_spr[4], req_vt[4];

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

static void uinit(const int me, const int pz, QCDMatrix_t u[4][LT2][LZ2][NY][NX])
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
	    int ix2 = ix % Nx2;
	    int iy2 = iy % Ny2;
	    int iz2 = ((iz-1)+((me%pz)*(LZ2-2))) % Nz2;
	    int it2 = ((it-1)+((me/pz)*(LT2-2))) % Nt2;
	    int ist2 = ix2 + Nx2*(iy2 + Ny2*(iz2 + Nz2*it2));
	    for(int ii = 0; ii < NCOL; ii++){
	      for(int jj = 0; jj < NCOL; jj++){
		int i2 = (ii*NCOL*2+jj*2) + NDF*(ist2+idir*Nst2);
		u[idir][it][iz][iy][ix].v[ii][jj][0] = (real_t)ur[i2];
		u[idir][it][iz][iy][ix].v[ii][jj][1] = (real_t)ur[i2+1];
	      }
	    }
	  }
	}
      }
    }
  }
  free(ur);
}

static void setconst(QCDSpinor_t v[LT2][LZ2][NY][NX], const real_t a)
{
  int it, iz, iy, ix, ii, jj, kk;
#pragma acc parallel loop collapse(7) present (v[0:LT2][:][:][:])
  for(it = 1; it < LT2-1; it++)
    for(iz = 1; iz < LZ2-1; iz++)
      for(iy = 0; iy < NY; iy++)
        for(ix = 0; ix < NX; ix++)
          for(ii = 0; ii < ND; ii++)
            for(jj = 0; jj < NCOL; jj++)
              for(kk = 0; kk < 2; kk++)
                v[it][iz][iy][ix].v[ii][jj][kk] = a;
}

static void set_src(const int me, const int ic, const int id, const int ix, const int iy, const int iz, const int it,
		    QCDSpinor_t v[LT2][LZ2][NY][NX])
{
  setconst(v, 0.0);

  if(me == 0){ // fix me
#pragma acc parallel present (v[0:LT2][:][:][:])
    v[it+1][iz+1][iy][ix].v[id][ic][0] = 1.0;
  }
}

static void test_mult(const int me, QCDMatrix_t u[4][LT2][LZ2][NY][NX])
{
  int nrepeat = 100;
  QCDSpinor_t bq2[LT2][LZ2][NY][NX], xq2[LT2][LZ2][NY][NX];
#pragma acc enter data create(bq2, xq2)
  set_src(me, 0, 0, 0, 0, 0, 0, bq2);

  int QCDSpinor_zyxvec = LZ * yx_Spinor;
#pragma acc host_data use_device(bq2, xq2)
  {
    MPI_Recv_init(&bq2[LT2-1][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, &req_w[2][0]);
    MPI_Recv_init(&bq2[0][1][0][0],     QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, &req_w[2][1]);
    MPI_Send_init(&bq2[1][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, &req_w[2][2]);
    MPI_Send_init(&bq2[LT2-2][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, &req_w[2][3]);
    
    MPI_Recv_init(&xq2[LT2-1][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, &req_w[3][0]);
    MPI_Recv_init(&xq2[0][1][0][0],     QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, &req_w[3][1]);
    MPI_Send_init(&xq2[1][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, &req_w[3][2]);
    MPI_Send_init(&xq2[LT2-2][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, &req_w[3][3]);
  }

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
#pragma acc exit data delete (bq2, xq2)
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

  //  acc_set_device_num(me%NGPUS+1, acc_device_nvidia);

  if(me == 0){
    printf("Simple Wilson solver\n\n");
    printf("NX = %3d, NY = %3d, NZ = %3d, NT = %3d\n", NX, NY, NZ, NT);
    printf("LX = %3d, LY = %3d, LZ = %3d, LT = %3d\n", NX, NY, LZ, LT);
    printf("(PT x PZ) = (%d x %d)\n", PT, PZ);
    printf("CKs = %10.6f\n", CKs);
    printf("enorm = %12.4e\n", enorm);
    printf("NUM=%d LEN=%d\n", NUM_GANGS, VECTOR_LENGTH);
  }

  int QCDSpinor_zyxvec = LZ * yx_Spinor;
  int QCDMatrix_zyxvec = LZ * yx_Matrix;
  int QCDSpinor_tyxvec = LT * yx_Spinor;
  int QCDMatrix_tyxvec = 4*LT*yx_Matrix;
  create_newcomm(PT, PZ, me);
  create_cart(PT, PZ, me);

  uinit(me, PZ, u);
#pragma acc enter data copyin(u)

#pragma acc host_data use_device(u, tmp_QCDMatrix_r, tmp_QCDMatrix_s, x, p, tmp_QCDSpinor_r, tmp_QCDSpinor_s, vt)
  {
    for(int i=0;i<4;i++){
      MPI_Recv_init(&u[i][0][1][0][0],     QCDMatrix_zyxvec, MPI_DOUBLE, left,  i, comm_lr, &req_u[i*2]);
      MPI_Send_init(&u[i][LT2-2][1][0][0], QCDMatrix_zyxvec, MPI_DOUBLE, right, i, comm_lr, &req_u[1+i*2]);
    }
    MPI_Recv_init(tmp_QCDMatrix_r, QCDMatrix_tyxvec, MPI_DOUBLE, up,   5, comm_ud, &req_mat[0]);
    MPI_Send_init(tmp_QCDMatrix_s, QCDMatrix_tyxvec, MPI_DOUBLE, down, 5, comm_ud, &req_mat[1]);
    
    //    MPI_Recv_init(x + ((LT2-1)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, &req_w[0][0]);
    //    MPI_Recv_init(x + yx_Spinor,                   QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, &req_w[0][1]);
    //    MPI_Send_init(x + (LZ2 + 1)*yx_Spinor,         QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, &req_w[0][2]);
    //    MPI_Send_init(x + ((LT2-2)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, &req_w[0][3]);
    //    MPI_Recv_init(p + ((LT2-1)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, &req_w[1][0]);
    //    MPI_Recv_init(p + yx_Spinor,                   QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, &req_w[1][1]);
    //    MPI_Send_init(p + (LZ2 + 1)*yx_Spinor,         QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, &req_w[1][2]);
    //    MPI_Send_init(p + ((LT2-2)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, &req_w[1][3]);

    MPI_Recv_init(&x[LT2-1][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, &req_w[0][0]);
    MPI_Recv_init(&x[0][1][0][0],     QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, &req_w[0][1]);
    MPI_Send_init(&x[1][1][0][0],     QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, &req_w[0][2]);
    MPI_Send_init(&x[LT2-2][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, &req_w[0][3]);

    MPI_Recv_init(&p[LT2-1][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, &req_w[1][0]);
    MPI_Recv_init(&p[0][1][0][0],     QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, &req_w[1][1]);
    MPI_Send_init(&p[1][1][0][0],     QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, &req_w[1][2]);
    MPI_Send_init(&p[LT2-2][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, &req_w[1][3]);

    MPI_Recv_init(&tmp_QCDSpinor_r[1][0][0][0], QCDSpinor_tyxvec, MPI_DOUBLE, down, 12, comm_ud, &req_spr[0]);
    MPI_Recv_init(&tmp_QCDSpinor_r[0][0][0][0], QCDSpinor_tyxvec, MPI_DOUBLE, up,   13, comm_ud, &req_spr[1]);
    MPI_Send_init(&tmp_QCDSpinor_s[0][0][0][0], QCDSpinor_tyxvec, MPI_DOUBLE, up,   12, comm_ud, &req_spr[2]);
    MPI_Send_init(&tmp_QCDSpinor_s[1][0][0][0], QCDSpinor_tyxvec, MPI_DOUBLE, down, 13, comm_ud, &req_spr[3]);
    
    //    MPI_Recv_init(vt + ((LT2-1)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, &req_vt[0]);
    //    MPI_Recv_init(vt + yx_Spinor,                   QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, &req_vt[1]);
    //    MPI_Send_init(vt + (LZ2 + 1)*yx_Spinor,         QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, &req_vt[2]);
    //    MPI_Send_init(vt + ((LT2-2)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, &req_vt[3]);
    MPI_Recv_init(&vt[LT2-1][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, &req_vt[0]);
    MPI_Recv_init(&vt[0][1][0][0],     QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, &req_vt[1]);
    MPI_Send_init(&vt[1][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, &req_vt[2]);
    MPI_Send_init(&vt[LT2-2][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, &req_vt[3]);
  }
  for(int it = 0; it < LT; it++)
    corr[it] = 0.0;

  test_mult(me, u);
#pragma acc enter data create(xq, bq)
  if(me == 0){
    printf("Solver:\n");
    printf("  ic  id   nconv      diff\n");
  }
  double time_tot = 0.0;
  double fop_tot  = 0.0;

  for(int ic = 0; ic < NCOL; ic++){
    for(int id = 0; id < ND; id++){
      set_src(me, ic, id, 0, 0, 0, 0, bq);
      MPI_Barrier(MPI_COMM_WORLD);
      double time0 = dtime();
      solve_CG(enorm, &nconv, &diff, xq, u, bq);
      MPI_Barrier(MPI_COMM_WORLD);
      double time1 = dtime();
      time_tot += time1 - time0;
      
      if(me == 0)
	printf(" %3d %3d  %6d %12.4e\n", ic, id, nconv, diff);
      
      double fop_mult1 = 2.0 * 1392.0 * (double)(NST);
      double fop_mult  = (double)(nconv+2) * fop_mult1;
      double fop_lin   = (double)(4+(nconv+1)*11) * (double)(NVST);
      fop_tot  += fop_lin + fop_mult;

      norm2_t(corr, xq);
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
