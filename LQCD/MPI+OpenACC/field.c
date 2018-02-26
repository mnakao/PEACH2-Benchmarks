/*
   Linear algebra.
                             [2016.05.03 Hideo Matsufuru]
*/
#include "lattice.h"

real_t dot(const QCDSpinor_t v1[LT2][LZ2][NY][NX], const QCDSpinor_t v2[LT2][LZ2][NY][NX])
{
  real_t a = 0.0;
  int it, iz, iy, ix, ii, jj, kk;
#pragma acc parallel loop reduction(+:a) collapse(7) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(v1[0:LT2][:][:][:], v2[0:LT2][:][:][:])
  for(it = 1; it < LT2-1; it++)
    for(iz = 1; iz < LZ2-1; iz++)
      for(iy = 0; iy < NY; iy++)
        for(ix = 0; ix < NX; ix++)
          for(ii = 0; ii < ND; ii++)
            for(jj = 0; jj < NCOL; jj++)
	      for(kk = 0; kk < 2; kk++)
		a += v1[it][iz][iy][ix].v[ii][jj][kk] * v2[it][iz][iy][ix].v[ii][jj][kk];

  MPI_Allreduce(MPI_IN_PLACE, &a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return a;
}

void norm2_t(real_t *restrict corr, const QCDSpinor_t v[LT2][LZ2][NY][NX])
{
  int it, iz, iy, ix, ii, jj, kk;

  for(it = 1; it < LT2-1; it++){
    real_t a = 0.0;
#pragma acc parallel loop reduction(+:a) collapse(6) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(v[0:LT2][:][:][:])
    for(iz = 1; iz < LZ2-1; iz++)
      for(iy = 0; iy < NY; iy++)
	for(ix = 0; ix < NX; ix++)
	  for(ii = 0; ii < ND; ii++)
	    for(jj = 0; jj < NCOL; jj++)
	      for(kk = 0; kk < 2; kk++)
		a += v[it][iz][iy][ix].v[ii][jj][kk] * v[it][iz][iy][ix].v[ii][jj][kk];

    corr[it-1] += a;
  }
}

real_t norm2(const QCDSpinor_t v[LT2][LZ2][NY][NX])
{
  real_t a = 0.0;
  int it, iz, iy, ix, ii, jj, kk;
#pragma acc parallel loop reduction(+:a) collapse(7) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(v[0:LT2][:][:][:])
  for(it = 1; it < LT2-1; it++)
    for(iz = 1; iz < LZ2-1; iz++)
      for(iy = 0; iy < NY; iy++)
	for(ix = 0; ix < NX; ix++)
	  for(ii = 0; ii < ND; ii++)
	    for(jj = 0; jj < NCOL; jj++)
	      for(kk = 0; kk < 2; kk++)
		a += v[it][iz][iy][ix].v[ii][jj][kk] * v[it][iz][iy][ix].v[ii][jj][kk];

  MPI_Allreduce(MPI_IN_PLACE, &a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return a;
}

void axpy(QCDSpinor_t v[LT2][LZ2][NY][NX], const real_t a, const QCDSpinor_t w[LT2][LZ2][NY][NX])
{
  int it, iz, iy, ix, ii, jj, kk;
#pragma acc parallel loop collapse(7) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(v[0:LT2][:][:][:], w[0:LT2][:][:][:])
  for(it = 1; it < LT2-1; it++)
    for(iz = 1; iz < LZ2-1; iz++)
      for(iy = 0; iy < NY; iy++)
        for(ix = 0; ix < NX; ix++)
          for(ii = 0; ii < ND; ii++)
            for(jj = 0; jj < NCOL; jj++)
	      for(kk = 0; kk < 2; kk++)
		v[it][iz][iy][ix].v[ii][jj][kk] += a * w[it][iz][iy][ix].v[ii][jj][kk];
}

void scal(QCDSpinor_t v[LT2][LZ2][NY][NX], const real_t a)
{
  int it, iz, iy, ix, ii, jj, kk;
#pragma acc parallel loop collapse(7) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(v[0:LT2][:][:][:])
  for(it = 1; it < LT2-1; it++)
    for(iz = 1; iz < LZ2-1; iz++)
      for(iy = 0; iy < NY; iy++)
        for(ix = 0; ix < NX; ix++)
          for(ii = 0; ii < ND; ii++)
            for(jj = 0; jj < NCOL; jj++)
	      for(kk = 0; kk < 2; kk++)
		v[it][iz][iy][ix].v[ii][jj][kk] *= a;
}

void copy(QCDSpinor_t v[LT2][LZ2][NY][NX], const QCDSpinor_t w[LT2][LZ2][NY][NX])
{
  int it, iz, iy, ix, ii, jj, kk;
#pragma acc parallel loop collapse(7) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(v[0:LT2][:][:][:], w[0:LT2][:][:][:])
  for(it = 1; it < LT2-1; it++)
    for(iz = 1; iz < LZ2-1; iz++)
      for(iy = 0; iy < NY; iy++)
	for(ix = 0; ix < NX; ix++)
	  for(ii = 0; ii < ND; ii++)
	    for(jj = 0; jj < NCOL; jj++)
	      for(kk = 0; kk < 2; kk++)
		v[it][iz][iy][ix].v[ii][jj][kk] = w[it][iz][iy][ix].v[ii][jj][kk];
}
