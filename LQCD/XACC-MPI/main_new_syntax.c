/* ****************************************************** */
/*    Wilson fermion solver in C language                 */
/*                                                        */
/*    OpenACC benchmark [5 May 0216 H.Matsufuru]          */
/*                                                        */
/*                     Copyright(c) Hideo Matsufuru 2016  */
/* ****************************************************** */
#include "lattice.h"

static QCDMatrix_t u[4][NT][NZ][NY][NX];
static QCDSpinor_t xq[NT][NZ][NY][NX], bq[NT][NZ][NY][NX];
#pragma xmp align u[*][i][j][*][*] with t(j,i) shadow 
#pragma xmp align xq[i][j][*][*] with t(j,i): xq, bq shadow

static real_t corr[NT];
#pragma xmp align corr[i] with t(*,i)

static real_t dot(const QCDSpinor_t v1[NT][NZ][NY][NX], const QCDSpinor_t v2[NT][NZ][NY][NX])
{
#pragma xmp align v1[i][j][*][*] with t(j,i): v1, v2 shadow
#pragma xmp static_desc :: v1, v2
  real_t a = 0.0;
  int it, iz, iy, ix, ii, jj, kk;

#pragma xmp loop (it, iz) on t(iz, it)
#pragma acc parallel loop reduction(+:a) collapse(7) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(v1, v2)
  for(it = 0; it < NT; it++)
    for(iz = 0; iz < NZ; iz++)
      for(iy = 0; iy < NY; iy++)
	for(ix = 0; ix < NX; ix++)
	  for(ii = 0; ii < ND; ii++)
	    for(jj = 0; jj < NCOL; jj++)
	      for(kk = 0; kk < 2; kk++)
		a += v1[it][iz][iy][ix].v[ii][jj][kk] * v2[it][iz][iy][ix].v[ii][jj][kk];

#pragma xmp reduction (+:a)
  return a;
}

static void norm2_t(real_t corr[NT], const QCDSpinor_t v[NT][NZ][NY][NX])
{
#pragma xmp align corr[i] with t(*,i)
#pragma xmp align v[i][j][*][*] with t(j,i): shadow
#pragma xmp static_desc :: v, corr
  int it, iz, iy, ix, ii, jj, kk;

#pragma xmp loop (it, iz) on t(iz, it)
  for(it = 0; it < NT; it++){
    real_t a = 0.0;
#pragma acc parallel loop reduction(+:a) collapse(6) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(v)
    for(iz = 0; iz < NZ; iz++)
      for(iy = 0; iy < NY; iy++)
	for(ix = 0; ix < NX; ix++)
	  for(ii = 0; ii < ND; ii++)
	    for(jj = 0; jj < NCOL; jj++)
	      for(kk = 0; kk < 2; kk++)
		a += v[it][iz][iy][ix].v[ii][jj][kk] * v[it][iz][iy][ix].v[ii][jj][kk];

    corr[it] += a;
  }
}

static real_t norm2(const QCDSpinor_t v[NT][NZ][NY][NX])
{
#pragma xmp align v[i][j][*][*] with t(j,i): v
#pragma xmp static_desc :: v
  real_t a = 0.0;
  int it, iz, iy, ix, ii, jj, kk;

#pragma xmp loop (it, iz) on t(iz, it)
#pragma acc parallel loop reduction(+:a) collapse(7) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(v)
  for(it = 0; it < NT; it++)
    for(iz = 0; iz < NZ; iz++)
      for(iy = 0; iy < NY; iy++)
	for(ix = 0; ix < NX; ix++)
	  for(ii = 0; ii < ND; ii++)
	    for(jj = 0; jj < NCOL; jj++)
	      for(kk = 0; kk < 2; kk++)
		a += v[it][iz][iy][ix].v[ii][jj][kk] * v[it][iz][iy][ix].v[ii][jj][kk];

#pragma xmp reduction (+:a)
  return a;
}

static void axpy(QCDSpinor_t v[NT][NZ][NY][NX], const real_t a, const QCDSpinor_t w[NT][NZ][NY][NX])
{
#pragma xmp align v[i][j][*][*] with t(j,i): v, w
#pragma xmp static_desc :: v, w
  int it, iz, iy, ix, ii, jj, kk;

#pragma xmp loop (it, iz) on t(iz, it)
#pragma acc parallel loop collapse(7) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(v, w)
  for(it = 0; it < NT; it++)
    for(iz = 0; iz < NZ; iz++)
      for(iy = 0; iy < NY; iy++)
	for(ix = 0; ix < NX; ix++)
	  for(ii = 0; ii < ND; ii++)
	    for(jj = 0; jj < NCOL; jj++)
	      for(kk = 0; kk < 2; kk++)
		v[it][iz][iy][ix].v[ii][jj][kk] += a * w[it][iz][iy][ix].v[ii][jj][kk];
}

static void scal(QCDSpinor_t v[NT][NZ][NY][NX], const real_t a)
{
#pragma xmp align v[i][j][*][*] with t(j,i): sha
#pragma xmp static_desc :: v
  int it, iz, iy, ix, ii, jj, kk;
  
#pragma xmp loop (it, iz) on t(iz, it)
#pragma acc parallel loop collapse(7) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(v)
  for(it = 0; it < NT; it++)
    for(iz = 0; iz < NZ; iz++)
      for(iy = 0; iy < NY; iy++)
        for(ix = 0; ix < NX; ix++)
          for(ii = 0; ii < ND; ii++)
            for(jj = 0; jj < NCOL; jj++)
	      for(kk = 0; kk < 2; kk++)
		v[it][iz][iy][ix].v[ii][jj][kk] *= a;
}

static void copy(QCDSpinor_t v[NT][NZ][NY][NX], const QCDSpinor_t w[NT][NZ][NY][NX])
{
#pragma xmp align v[i][j][*][*] with t(j,i): v shadow
#pragma xmp static_desc :: v, w
  int it, iz, iy, ix, ii, jj, kk;

#pragma xmp loop (it, iz) on t(iz, it)
#pragma acc parallel loop collapse(7) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(v, w)
  for(it = 0; it < NT; it++)
    for(iz = 0; iz < NZ; iz++)
      for(iy = 0; iy < NY; iy++)
	for(ix = 0; ix < NX; ix++)
	  for(ii = 0; ii < ND; ii++)
	    for(jj = 0; jj < NCOL; jj++)
	      for(kk = 0; kk < 2; kk++)
		v[it][iz][iy][ix].v[ii][jj][kk] = w[it][iz][iy][ix].v[ii][jj][kk];
}

static void opr_H_alt(QCDSpinor_t v2[NT][NZ][NY][NX], QCDMatrix_t u[4][NT][NZ][NY][NX],
		      QCDSpinor_t v1[NT][NZ][NY][NX])
{
#pragma xmp align v2[i][j][*][*] with t(j,i): v2, v1 shadow
#pragma xmp align u[*][i][j][*][*] with t(j,i) shadow
#pragma xmp static_desc :: u, v1, v2
  static QCDSpinor_t v2L;
#pragma acc enter data pcreate(v2L)
  int it, iz, iy, ix;

#pragma xmp loop (iz,it) on t(iz,it)
#pragma acc parallel loop collapse(4) private(v2L) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(v2, u, v1, v2L)
  for(it = 0; it < NT; it++){
    for(iz = 0; iz < NZ; iz++){
      for(iy = 0; iy < NY; iy++){
	for(ix = 0; ix < NX; ix++){
	  real_t u_0, u_1, u_2, u_3, u_4, u_5;
	  real_t u_6, u_7, u_8, u_9, u10, u11;
	  real_t u12, u13, u14, u15, u16, u17;
	  real_t vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5;
	  real_t vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5;
	  real_t wt1r, wt1i, wt2r, wt2i;

	  // mult_xp
	  int idir = 0;
	  int ic;
	  int nn = (ix + 1) % NX; // (ix == NX - 1)? 0 : ix + 1;

	  vt1_0 = v1[it][iz][iy][nn].v[0][0][0] - v1[it][iz][iy][nn].v[3][0][1];
	  vt1_1 = v1[it][iz][iy][nn].v[0][0][1] + v1[it][iz][iy][nn].v[3][0][0];
	  vt1_2 = v1[it][iz][iy][nn].v[0][1][0] - v1[it][iz][iy][nn].v[3][1][1];
	  vt1_3 = v1[it][iz][iy][nn].v[0][1][1] + v1[it][iz][iy][nn].v[3][1][0];
	  vt1_4 = v1[it][iz][iy][nn].v[0][2][0] - v1[it][iz][iy][nn].v[3][2][1];
	  vt1_5 = v1[it][iz][iy][nn].v[0][2][1] + v1[it][iz][iy][nn].v[3][2][0];

	  vt2_0 = v1[it][iz][iy][nn].v[1][0][0] - v1[it][iz][iy][nn].v[2][0][1];
          vt2_1 = v1[it][iz][iy][nn].v[1][0][1] + v1[it][iz][iy][nn].v[2][0][0];
          vt2_2 = v1[it][iz][iy][nn].v[1][1][0] - v1[it][iz][iy][nn].v[2][1][1];
          vt2_3 = v1[it][iz][iy][nn].v[1][1][1] + v1[it][iz][iy][nn].v[2][1][0];
          vt2_4 = v1[it][iz][iy][nn].v[1][2][0] - v1[it][iz][iy][nn].v[2][2][1];
          vt2_5 = v1[it][iz][iy][nn].v[1][2][1] + v1[it][iz][iy][nn].v[2][2][0];
    
	  u_0 = u[idir][it][iz][iy][ix].v[0][0][0];
	  u_1 = u[idir][it][iz][iy][ix].v[0][0][1];
	  u_2 = u[idir][it][iz][iy][ix].v[0][1][0];
	  u_3 = u[idir][it][iz][iy][ix].v[0][1][1];
	  u_4 = u[idir][it][iz][iy][ix].v[0][2][0];
	  u_5 = u[idir][it][iz][iy][ix].v[0][2][1];

	  wt1r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

	  ic = 0;
	  v2L.v[0][ic][0] =  wt1r; v2L.v[0][ic][1] =  wt1i;
	  v2L.v[1][ic][0] =  wt2r; v2L.v[1][ic][1] =  wt2i;
	  v2L.v[2][ic][0] =  wt2i; v2L.v[2][ic][1] = -wt2r;
	  v2L.v[3][ic][0] =  wt1i; v2L.v[3][ic][1] = -wt1r;

	  u_6 = u[idir][it][iz][iy][ix].v[1][0][0];
          u_7 = u[idir][it][iz][iy][ix].v[1][0][1];
          u_8 = u[idir][it][iz][iy][ix].v[1][1][0];
          u_9 = u[idir][it][iz][iy][ix].v[1][1][1];
          u10 = u[idir][it][iz][iy][ix].v[1][2][0];
          u11 = u[idir][it][iz][iy][ix].v[1][2][1];

	  wt1r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

	  ic = 1;
	  v2L.v[0][ic][0] =  wt1r; v2L.v[0][ic][1] =  wt1i;
	  v2L.v[1][ic][0] =  wt2r; v2L.v[1][ic][1] =  wt2i;
	  v2L.v[2][ic][0] =  wt2i; v2L.v[2][ic][1] = -wt2r;
	  v2L.v[3][ic][0] =  wt1i; v2L.v[3][ic][1] = -wt1r;

	  u12 = EXT_IMG_R(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u13 = EXT_IMG_I(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u14 = EXT_IMG_R(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u15 = EXT_IMG_I(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u16 = EXT_IMG_R(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);
	  u17 = EXT_IMG_I(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);

	  wt1r = MULT_GXr(u12, u13, u14, u15, u16, u17,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GXi(u12, u13, u14, u15, u16, u17,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GXr(u12, u13, u14, u15, u16, u17,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GXi(u12, u13, u14, u15, u16, u17,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
	  ic = 2;
	  v2L.v[0][ic][0] =  wt1r; v2L.v[0][ic][1] =  wt1i;
	  v2L.v[1][ic][0] =  wt2r; v2L.v[1][ic][1] =  wt2i;
	  v2L.v[2][ic][0] =  wt2i; v2L.v[2][ic][1] = -wt2r;
	  v2L.v[3][ic][0] =  wt1i; v2L.v[3][ic][1] = -wt1r;

	  // mult_xm
	  nn = (ix + NX - 1) % NX; // (ix == 0)? NX - 1 : ix - 1;

	  vt1_0 = v1[it][iz][iy][nn].v[0][0][0] + v1[it][iz][iy][nn].v[3][0][1];
	  vt1_1 = v1[it][iz][iy][nn].v[0][0][1] - v1[it][iz][iy][nn].v[3][0][0];
	  vt1_2 = v1[it][iz][iy][nn].v[0][1][0] + v1[it][iz][iy][nn].v[3][1][1];
	  vt1_3 = v1[it][iz][iy][nn].v[0][1][1] - v1[it][iz][iy][nn].v[3][1][0];
	  vt1_4 = v1[it][iz][iy][nn].v[0][2][0] + v1[it][iz][iy][nn].v[3][2][1];
	  vt1_5 = v1[it][iz][iy][nn].v[0][2][1] - v1[it][iz][iy][nn].v[3][2][0];
    
	  vt2_0 = v1[it][iz][iy][nn].v[1][0][0] + v1[it][iz][iy][nn].v[2][0][1];
	  vt2_1 = v1[it][iz][iy][nn].v[1][0][1] - v1[it][iz][iy][nn].v[2][0][0];
	  vt2_2 = v1[it][iz][iy][nn].v[1][1][0] + v1[it][iz][iy][nn].v[2][1][1];
	  vt2_3 = v1[it][iz][iy][nn].v[1][1][1] - v1[it][iz][iy][nn].v[2][1][0];
	  vt2_4 = v1[it][iz][iy][nn].v[1][2][0] + v1[it][iz][iy][nn].v[2][2][1];
	  vt2_5 = v1[it][iz][iy][nn].v[1][2][1] - v1[it][iz][iy][nn].v[2][2][0];
    
	  u_0 = u[idir][it][iz][iy][nn].v[0][0][0];
	  u_1 = u[idir][it][iz][iy][nn].v[0][0][1];
	  u_2 = u[idir][it][iz][iy][nn].v[1][0][0];
	  u_3 = u[idir][it][iz][iy][nn].v[1][0][1];
	  u_4 = u[idir][it][iz][iy][nn].v[2][0][0];
	  u_5 = u[idir][it][iz][iy][nn].v[2][0][1];
	  
	  wt1r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
	  ic = 0;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  v2L.v[2][ic][0] += -wt2i; v2L.v[2][ic][1] +=  wt2r;
	  v2L.v[3][ic][0] += -wt1i; v2L.v[3][ic][1] +=  wt1r;

	  u_6 = u[idir][it][iz][iy][nn].v[0][1][0];
	  u_7 = u[idir][it][iz][iy][nn].v[0][1][1];
	  u_8 = u[idir][it][iz][iy][nn].v[1][1][0];
	  u_9 = u[idir][it][iz][iy][nn].v[1][1][1];
	  u10 = u[idir][it][iz][iy][nn].v[2][1][0];
	  u11 = u[idir][it][iz][iy][nn].v[2][1][1];
    
	  wt1r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
	  ic = 1;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  v2L.v[2][ic][0] += -wt2i; v2L.v[2][ic][1] +=  wt2r;
	  v2L.v[3][ic][0] += -wt1i; v2L.v[3][ic][1] +=  wt1r;
    
	  u12 = EXT_IMG_R(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u13 = EXT_IMG_I(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u14 = EXT_IMG_R(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u15 = EXT_IMG_I(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u16 = EXT_IMG_R(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);
	  u17 = EXT_IMG_I(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);
    
	  wt1r = MULT_GDXr(u12, u13, u14, u15, u16, u17,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GDXi(u12, u13, u14, u15, u16, u17,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GDXr(u12, u13, u14, u15, u16, u17,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GDXi(u12, u13, u14, u15, u16, u17,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
	  ic = 2;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  v2L.v[2][ic][0] += -wt2i; v2L.v[2][ic][1] +=  wt2r;
	  v2L.v[3][ic][0] += -wt1i; v2L.v[3][ic][1] +=  wt1r;

	  // mult_yp
	  nn = (iy + 1) % NY; // (iy == NY - 1)? 0 : iy + 1;
	  idir = 1;

	  vt1_0 = v1[it][iz][nn][ix].v[0][0][0] + v1[it][iz][nn][ix].v[3][0][0];
	  vt1_1 = v1[it][iz][nn][ix].v[0][0][1] + v1[it][iz][nn][ix].v[3][0][1];
	  vt1_2 = v1[it][iz][nn][ix].v[0][1][0] + v1[it][iz][nn][ix].v[3][1][0];
	  vt1_3 = v1[it][iz][nn][ix].v[0][1][1] + v1[it][iz][nn][ix].v[3][1][1];
	  vt1_4 = v1[it][iz][nn][ix].v[0][2][0] + v1[it][iz][nn][ix].v[3][2][0];
	  vt1_5 = v1[it][iz][nn][ix].v[0][2][1] + v1[it][iz][nn][ix].v[3][2][1];
	  
	  vt2_0 = v1[it][iz][nn][ix].v[1][0][0] - v1[it][iz][nn][ix].v[2][0][0];
	  vt2_1 = v1[it][iz][nn][ix].v[1][0][1] - v1[it][iz][nn][ix].v[2][0][1];
	  vt2_2 = v1[it][iz][nn][ix].v[1][1][0] - v1[it][iz][nn][ix].v[2][1][0];
	  vt2_3 = v1[it][iz][nn][ix].v[1][1][1] - v1[it][iz][nn][ix].v[2][1][1];
	  vt2_4 = v1[it][iz][nn][ix].v[1][2][0] - v1[it][iz][nn][ix].v[2][2][0];
	  vt2_5 = v1[it][iz][nn][ix].v[1][2][1] - v1[it][iz][nn][ix].v[2][2][1];
    
	  u_0 = u[idir][it][iz][iy][ix].v[0][0][0];
	  u_1 = u[idir][it][iz][iy][ix].v[0][0][1];
	  u_2 = u[idir][it][iz][iy][ix].v[0][1][0];
	  u_3 = u[idir][it][iz][iy][ix].v[0][1][1];
	  u_4 = u[idir][it][iz][iy][ix].v[0][2][0];
	  u_5 = u[idir][it][iz][iy][ix].v[0][2][1];
    
	  wt1r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
	  ic = 0;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  v2L.v[2][ic][0] += -wt2r; v2L.v[2][ic][1] += -wt2i;
	  v2L.v[3][ic][0] +=  wt1r; v2L.v[3][ic][1] +=  wt1i;

	  u_6 = u[idir][it][iz][iy][ix].v[1][0][0];
	  u_7 = u[idir][it][iz][iy][ix].v[1][0][1];
	  u_8 = u[idir][it][iz][iy][ix].v[1][1][0];
	  u_9 = u[idir][it][iz][iy][ix].v[1][1][1];
	  u10 = u[idir][it][iz][iy][ix].v[1][2][0];
	  u11 = u[idir][it][iz][iy][ix].v[1][2][1];
    
	  wt1r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  
	  ic = 1;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  v2L.v[2][ic][0] += -wt2r; v2L.v[2][ic][1] += -wt2i;
	  v2L.v[3][ic][0] +=  wt1r; v2L.v[3][ic][1] +=  wt1i;
    
	  u12 = EXT_IMG_R(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u13 = EXT_IMG_I(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u14 = EXT_IMG_R(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u15 = EXT_IMG_I(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u16 = EXT_IMG_R(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);
	  u17 = EXT_IMG_I(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);
    
	  wt1r = MULT_GXr(u12, u13, u14, u15, u16, u17,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GXi(u12, u13, u14, u15, u16, u17,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GXr(u12, u13, u14, u15, u16, u17,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GXi(u12, u13, u14, u15, u16, u17,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
	  ic = 2;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  v2L.v[2][ic][0] += -wt2r; v2L.v[2][ic][1] += -wt2i;
	  v2L.v[3][ic][0] +=  wt1r; v2L.v[3][ic][1] +=  wt1i;

	  // mult_ym
	  nn = (iy + NY - 1) % NY; // (iy == 0)? NY - 1 : iy - 1;

	  vt1_0 = v1[it][iz][nn][ix].v[0][0][0] - v1[it][iz][nn][ix].v[3][0][0];
	  vt1_1 = v1[it][iz][nn][ix].v[0][0][1] - v1[it][iz][nn][ix].v[3][0][1];
	  vt1_2 = v1[it][iz][nn][ix].v[0][1][0] - v1[it][iz][nn][ix].v[3][1][0];
	  vt1_3 = v1[it][iz][nn][ix].v[0][1][1] - v1[it][iz][nn][ix].v[3][1][1];
	  vt1_4 = v1[it][iz][nn][ix].v[0][2][0] - v1[it][iz][nn][ix].v[3][2][0];
	  vt1_5 = v1[it][iz][nn][ix].v[0][2][1] - v1[it][iz][nn][ix].v[3][2][1];
    
	  vt2_0 = v1[it][iz][nn][ix].v[1][0][0] + v1[it][iz][nn][ix].v[2][0][0];
	  vt2_1 = v1[it][iz][nn][ix].v[1][0][1] + v1[it][iz][nn][ix].v[2][0][1];
	  vt2_2 = v1[it][iz][nn][ix].v[1][1][0] + v1[it][iz][nn][ix].v[2][1][0];
	  vt2_3 = v1[it][iz][nn][ix].v[1][1][1] + v1[it][iz][nn][ix].v[2][1][1];
	  vt2_4 = v1[it][iz][nn][ix].v[1][2][0] + v1[it][iz][nn][ix].v[2][2][0];
	  vt2_5 = v1[it][iz][nn][ix].v[1][2][1] + v1[it][iz][nn][ix].v[2][2][1];
    
	  u_0 = u[idir][it][iz][nn][ix].v[0][0][0];
	  u_1 = u[idir][it][iz][nn][ix].v[0][0][1];
	  u_2 = u[idir][it][iz][nn][ix].v[1][0][0];
	  u_3 = u[idir][it][iz][nn][ix].v[1][0][1];
	  u_4 = u[idir][it][iz][nn][ix].v[2][0][0];
	  u_5 = u[idir][it][iz][nn][ix].v[2][0][1];

	  wt1r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  
	  ic = 0;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  v2L.v[2][ic][0] +=  wt2r; v2L.v[2][ic][1] +=  wt2i;
	  v2L.v[3][ic][0] += -wt1r; v2L.v[3][ic][1] += -wt1i;
    
	  u_6 = u[idir][it][iz][nn][ix].v[0][1][0];
	  u_7 = u[idir][it][iz][nn][ix].v[0][1][1];
	  u_8 = u[idir][it][iz][nn][ix].v[1][1][0];
	  u_9 = u[idir][it][iz][nn][ix].v[1][1][1];
	  u10 = u[idir][it][iz][nn][ix].v[2][1][0];
	  u11 = u[idir][it][iz][nn][ix].v[2][1][1];
    
	  wt1r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  
	  ic = 1;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  v2L.v[2][ic][0] +=  wt2r; v2L.v[2][ic][1] +=  wt2i;
	  v2L.v[3][ic][0] += -wt1r; v2L.v[3][ic][1] += -wt1i;
    
	  u12 = EXT_IMG_R(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u13 = EXT_IMG_I(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u14 = EXT_IMG_R(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u15 = EXT_IMG_I(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u16 = EXT_IMG_R(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);
	  u17 = EXT_IMG_I(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);
	  
	  wt1r = MULT_GDXr(u12, u13, u14, u15, u16, u17,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GDXi(u12, u13, u14, u15, u16, u17,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GDXr(u12, u13, u14, u15, u16, u17,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GDXi(u12, u13, u14, u15, u16, u17,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
	  ic = 2;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  v2L.v[2][ic][0] +=  wt2r; v2L.v[2][ic][1] +=  wt2i;
	  v2L.v[3][ic][0] += -wt1r; v2L.v[3][ic][1] += -wt1i;

	  // mult_zp
	  idir = 2;
	  nn = iz + 1;	  // nn = (iz + 1) % NZ; // (iz == NZ - 1)? 0 : iz + 1;

	  vt1_0 = v1[it][iz+1][iy][ix].v[0][0][0] - v1[it][iz+1][iy][ix].v[2][0][1];
	  vt1_1 = v1[it][iz+1][iy][ix].v[0][0][1] + v1[it][iz+1][iy][ix].v[2][0][0];
	  vt1_2 = v1[it][iz+1][iy][ix].v[0][1][0] - v1[it][iz+1][iy][ix].v[2][1][1];
	  vt1_3 = v1[it][iz+1][iy][ix].v[0][1][1] + v1[it][iz+1][iy][ix].v[2][1][0];
	  vt1_4 = v1[it][iz+1][iy][ix].v[0][2][0] - v1[it][iz+1][iy][ix].v[2][2][1];
	  vt1_5 = v1[it][iz+1][iy][ix].v[0][2][1] + v1[it][iz+1][iy][ix].v[2][2][0];
    
	  vt2_0 = v1[it][iz+1][iy][ix].v[1][0][0] + v1[it][iz+1][iy][ix].v[3][0][1];
	  vt2_1 = v1[it][iz+1][iy][ix].v[1][0][1] - v1[it][iz+1][iy][ix].v[3][0][0];
	  vt2_2 = v1[it][iz+1][iy][ix].v[1][1][0] + v1[it][iz+1][iy][ix].v[3][1][1];
	  vt2_3 = v1[it][iz+1][iy][ix].v[1][1][1] - v1[it][iz+1][iy][ix].v[3][1][0];
	  vt2_4 = v1[it][iz+1][iy][ix].v[1][2][0] + v1[it][iz+1][iy][ix].v[3][2][1];
	  vt2_5 = v1[it][iz+1][iy][ix].v[1][2][1] - v1[it][iz+1][iy][ix].v[3][2][0];
    
	  u_0 = u[idir][it][iz][iy][ix].v[0][0][0];
	  u_1 = u[idir][it][iz][iy][ix].v[0][0][1];
	  u_2 = u[idir][it][iz][iy][ix].v[0][1][0];
	  u_3 = u[idir][it][iz][iy][ix].v[0][1][1];
	  u_4 = u[idir][it][iz][iy][ix].v[0][2][0];
	  u_5 = u[idir][it][iz][iy][ix].v[0][2][1];
    
	  wt1r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  
	  ic = 0;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  v2L.v[2][ic][0] +=  wt1i; v2L.v[2][ic][1] += -wt1r;
	  v2L.v[3][ic][0] += -wt2i; v2L.v[3][ic][1] +=  wt2r;

	  u_6 = u[idir][it][iz][iy][ix].v[1][0][0];
	  u_7 = u[idir][it][iz][iy][ix].v[1][0][1];
	  u_8 = u[idir][it][iz][iy][ix].v[1][1][0];
	  u_9 = u[idir][it][iz][iy][ix].v[1][1][1];
	  u10 = u[idir][it][iz][iy][ix].v[1][2][0];
	  u11 = u[idir][it][iz][iy][ix].v[1][2][1];
    
	  wt1r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

	  ic = 1;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  v2L.v[2][ic][0] +=  wt1i; v2L.v[2][ic][1] += -wt1r;
	  v2L.v[3][ic][0] += -wt2i; v2L.v[3][ic][1] +=  wt2r;
    
	  u12 = EXT_IMG_R(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u13 = EXT_IMG_I(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u14 = EXT_IMG_R(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u15 = EXT_IMG_I(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u16 = EXT_IMG_R(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);
	  u17 = EXT_IMG_I(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);
    
	  wt1r = MULT_GXr(u12, u13, u14, u15, u16, u17,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GXi(u12, u13, u14, u15, u16, u17,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GXr(u12, u13, u14, u15, u16, u17,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GXi(u12, u13, u14, u15, u16, u17,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
	  ic = 2;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  v2L.v[2][ic][0] +=  wt1i; v2L.v[2][ic][1] += -wt1r;
	  v2L.v[3][ic][0] += -wt2i; v2L.v[3][ic][1] +=  wt2r;

	  // mult_zm
	  nn = (iz + NZ - 1) % NZ; // (iz == 0)? NZ - 1 : iz - 1;

	  vt1_0 = v1[it][iz-1][iy][ix].v[0][0][0] + v1[it][iz-1][iy][ix].v[2][0][1];
	  vt1_1 = v1[it][iz-1][iy][ix].v[0][0][1] - v1[it][iz-1][iy][ix].v[2][0][0];
	  vt1_2 = v1[it][iz-1][iy][ix].v[0][1][0] + v1[it][iz-1][iy][ix].v[2][1][1];
	  vt1_3 = v1[it][iz-1][iy][ix].v[0][1][1] - v1[it][iz-1][iy][ix].v[2][1][0];
	  vt1_4 = v1[it][iz-1][iy][ix].v[0][2][0] + v1[it][iz-1][iy][ix].v[2][2][1];
	  vt1_5 = v1[it][iz-1][iy][ix].v[0][2][1] - v1[it][iz-1][iy][ix].v[2][2][0];
    
	  vt2_0 = v1[it][iz-1][iy][ix].v[1][0][0] - v1[it][iz-1][iy][ix].v[3][0][1];
	  vt2_1 = v1[it][iz-1][iy][ix].v[1][0][1] + v1[it][iz-1][iy][ix].v[3][0][0];
	  vt2_2 = v1[it][iz-1][iy][ix].v[1][1][0] - v1[it][iz-1][iy][ix].v[3][1][1];
	  vt2_3 = v1[it][iz-1][iy][ix].v[1][1][1] + v1[it][iz-1][iy][ix].v[3][1][0];
	  vt2_4 = v1[it][iz-1][iy][ix].v[1][2][0] - v1[it][iz-1][iy][ix].v[3][2][1];
	  vt2_5 = v1[it][iz-1][iy][ix].v[1][2][1] + v1[it][iz-1][iy][ix].v[3][2][0];
    
	  u_0 = u[idir][it][iz-1][iy][ix].v[0][0][0];
	  u_1 = u[idir][it][iz-1][iy][ix].v[0][0][1];
	  u_2 = u[idir][it][iz-1][iy][ix].v[1][0][0];
	  u_3 = u[idir][it][iz-1][iy][ix].v[1][0][1];
	  u_4 = u[idir][it][iz-1][iy][ix].v[2][0][0];
	  u_5 = u[idir][it][iz-1][iy][ix].v[2][0][1];
    
	  wt1r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
	  ic = 0;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  v2L.v[2][ic][0] += -wt1i; v2L.v[2][ic][1] +=  wt1r;
	  v2L.v[3][ic][0] +=  wt2i; v2L.v[3][ic][1] += -wt2r;
    
	  u_6 = u[idir][it][iz-1][iy][ix].v[0][1][0];
	  u_7 = u[idir][it][iz-1][iy][ix].v[0][1][1];
	  u_8 = u[idir][it][iz-1][iy][ix].v[1][1][0];
	  u_9 = u[idir][it][iz-1][iy][ix].v[1][1][1];
	  u10 = u[idir][it][iz-1][iy][ix].v[2][1][0];
	  u11 = u[idir][it][iz-1][iy][ix].v[2][1][1];
    
	  wt1r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

	  ic = 1;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  v2L.v[2][ic][0] += -wt1i; v2L.v[2][ic][1] +=  wt1r;
	  v2L.v[3][ic][0] +=  wt2i; v2L.v[3][ic][1] += -wt2r;
    
	  u12 = EXT_IMG_R(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u13 = EXT_IMG_I(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u14 = EXT_IMG_R(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u15 = EXT_IMG_I(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u16 = EXT_IMG_R(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);
	  u17 = EXT_IMG_I(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);
    
	  wt1r = MULT_GDXr(u12, u13, u14, u15, u16, u17,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GDXi(u12, u13, u14, u15, u16, u17,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GDXr(u12, u13, u14, u15, u16, u17,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GDXi(u12, u13, u14, u15, u16, u17,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  
	  ic = 2;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  v2L.v[2][ic][0] += -wt1i; v2L.v[2][ic][1] +=  wt1r;
	  v2L.v[3][ic][0] +=  wt2i; v2L.v[3][ic][1] += -wt2r;

	  // mult_tp
	  idir = 3;
	  nn = it + 1; // nn = (it + 1) % NT;

	  vt1_0 = 2.0 * v1[it+1][iz][iy][ix].v[2][0][0];
	  vt1_1 = 2.0 * v1[it+1][iz][iy][ix].v[2][0][1];
	  vt1_2 = 2.0 * v1[it+1][iz][iy][ix].v[2][1][0];
	  vt1_3 = 2.0 * v1[it+1][iz][iy][ix].v[2][1][1];
	  vt1_4 = 2.0 * v1[it+1][iz][iy][ix].v[2][2][0];
	  vt1_5 = 2.0 * v1[it+1][iz][iy][ix].v[2][2][1];

	  vt2_0 = 2.0 * v1[it+1][iz][iy][ix].v[3][0][0];
	  vt2_1 = 2.0 * v1[it+1][iz][iy][ix].v[3][0][1];
	  vt2_2 = 2.0 * v1[it+1][iz][iy][ix].v[3][1][0];
	  vt2_3 = 2.0 * v1[it+1][iz][iy][ix].v[3][1][1];
	  vt2_4 = 2.0 * v1[it+1][iz][iy][ix].v[3][2][0];
	  vt2_5 = 2.0 * v1[it+1][iz][iy][ix].v[3][2][1];

	  u_0 = u[idir][it][iz][iy][ix].v[0][0][0];
	  u_1 = u[idir][it][iz][iy][ix].v[0][0][1];
	  u_2 = u[idir][it][iz][iy][ix].v[0][1][0];
	  u_3 = u[idir][it][iz][iy][ix].v[0][1][1];
	  u_4 = u[idir][it][iz][iy][ix].v[0][2][0];
	  u_5 = u[idir][it][iz][iy][ix].v[0][2][1];

	  wt1r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

	  ic = 0;
	  v2L.v[2][ic][0] +=  wt1r; v2L.v[2][ic][1] +=  wt1i;
	  v2L.v[3][ic][0] +=  wt2r; v2L.v[3][ic][1] +=  wt2i;

	  u_6 = u[idir][it][iz][iy][ix].v[1][0][0];
	  u_7 = u[idir][it][iz][iy][ix].v[1][0][1];
	  u_8 = u[idir][it][iz][iy][ix].v[1][1][0];
	  u_9 = u[idir][it][iz][iy][ix].v[1][1][1];
	  u10 = u[idir][it][iz][iy][ix].v[1][2][0];
	  u11 = u[idir][it][iz][iy][ix].v[1][2][1];
    
	  wt1r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

	  ic = 1;
	  v2L.v[2][ic][0] +=  wt1r; v2L.v[2][ic][1] +=  wt1i;
	  v2L.v[3][ic][0] +=  wt2r; v2L.v[3][ic][1] +=  wt2i;
    
	  u12 = EXT_IMG_R(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u13 = EXT_IMG_I(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u14 = EXT_IMG_R(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u15 = EXT_IMG_I(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u16 = EXT_IMG_R(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);
	  u17 = EXT_IMG_I(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);
    
	  wt1r = MULT_GXr(u12, u13, u14, u15, u16, u17,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GXi(u12, u13, u14, u15, u16, u17,
			  vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GXr(u12, u13, u14, u15, u16, u17,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GXi(u12, u13, u14, u15, u16, u17,
			  vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  
	  ic = 2;
	  v2L.v[2][ic][0] +=  wt1r; v2L.v[2][ic][1] +=  wt1i;
	  v2L.v[3][ic][0] +=  wt2r; v2L.v[3][ic][1] +=  wt2i;

	  // mult_tm
	  nn = it - 1; // nn = (it + NT - 1) % NT;

	  vt1_0 = 2.0 * v1[it-1][iz][iy][ix].v[0][0][0];
	  vt1_1 = 2.0 * v1[it-1][iz][iy][ix].v[0][0][1];
	  vt1_2 = 2.0 * v1[it-1][iz][iy][ix].v[0][1][0];
	  vt1_3 = 2.0 * v1[it-1][iz][iy][ix].v[0][1][1];
	  vt1_4 = 2.0 * v1[it-1][iz][iy][ix].v[0][2][0];
	  vt1_5 = 2.0 * v1[it-1][iz][iy][ix].v[0][2][1];

	  vt2_0 = 2.0 * v1[it-1][iz][iy][ix].v[1][0][0];
	  vt2_1 = 2.0 * v1[it-1][iz][iy][ix].v[1][0][1];
	  vt2_2 = 2.0 * v1[it-1][iz][iy][ix].v[1][1][0];
	  vt2_3 = 2.0 * v1[it-1][iz][iy][ix].v[1][1][1];
	  vt2_4 = 2.0 * v1[it-1][iz][iy][ix].v[1][2][0];
	  vt2_5 = 2.0 * v1[it-1][iz][iy][ix].v[1][2][1];
    
	  u_0 = u[idir][it-1][iz][iy][ix].v[0][0][0];
	  u_1 = u[idir][it-1][iz][iy][ix].v[0][0][1];
	  u_2 = u[idir][it-1][iz][iy][ix].v[1][0][0];
	  u_3 = u[idir][it-1][iz][iy][ix].v[1][0][1];
	  u_4 = u[idir][it-1][iz][iy][ix].v[2][0][0];
	  u_5 = u[idir][it-1][iz][iy][ix].v[2][0][1];
    
	  wt1r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
	  ic = 0;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
    
	  u_6 = u[idir][it-1][iz][iy][ix].v[0][1][0];
	  u_7 = u[idir][it-1][iz][iy][ix].v[0][1][1];
	  u_8 = u[idir][it-1][iz][iy][ix].v[1][1][0];
	  u_9 = u[idir][it-1][iz][iy][ix].v[1][1][1];
	  u10 = u[idir][it-1][iz][iy][ix].v[2][1][0];
	  u11 = u[idir][it-1][iz][iy][ix].v[2][1][1];
    
	  wt1r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
	  ic = 1;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;
	  
	  u12 = EXT_IMG_R(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u13 = EXT_IMG_I(u_2, u_3, u_4, u_5, u_8, u_9, u10, u11);
	  u14 = EXT_IMG_R(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u15 = EXT_IMG_I(u_4, u_5, u_0, u_1, u10, u11, u_6, u_7);
	  u16 = EXT_IMG_R(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);
	  u17 = EXT_IMG_I(u_0, u_1, u_2, u_3, u_6, u_7, u_8, u_9);
	  
	  wt1r = MULT_GDXr(u12, u13, u14, u15, u16, u17,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt1i = MULT_GDXi(u12, u13, u14, u15, u16, u17,
			   vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
	  wt2r = MULT_GDXr(u12, u13, u14, u15, u16, u17,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  wt2i = MULT_GDXi(u12, u13, u14, u15, u16, u17,
			   vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
	  
	  ic = 2;
	  v2L.v[0][ic][0] +=  wt1r; v2L.v[0][ic][1] +=  wt1i;
	  v2L.v[1][ic][0] +=  wt2r; v2L.v[1][ic][1] +=  wt2i;

	  // gm5_aypx and write back to global memory
	  v2[it][iz][iy][ix].v[2][0][0] = v1[it][iz][iy][ix].v[0][0][0] - CKs * v2L.v[0][0][0];
	  v2[it][iz][iy][ix].v[2][0][1] = v1[it][iz][iy][ix].v[0][0][1] - CKs * v2L.v[0][0][1];
	  v2[it][iz][iy][ix].v[2][1][0] = v1[it][iz][iy][ix].v[0][1][0] - CKs * v2L.v[0][1][0];
	  v2[it][iz][iy][ix].v[2][1][1] = v1[it][iz][iy][ix].v[0][1][1] - CKs * v2L.v[0][1][1];
	  v2[it][iz][iy][ix].v[2][2][0] = v1[it][iz][iy][ix].v[0][2][0] - CKs * v2L.v[0][2][0];
	  v2[it][iz][iy][ix].v[2][2][1] = v1[it][iz][iy][ix].v[0][2][1] - CKs * v2L.v[0][2][1];

	  v2[it][iz][iy][ix].v[3][0][0] = v1[it][iz][iy][ix].v[1][0][0] - CKs * v2L.v[1][0][0];
          v2[it][iz][iy][ix].v[3][0][1] = v1[it][iz][iy][ix].v[1][0][1] - CKs * v2L.v[1][0][1];
	  v2[it][iz][iy][ix].v[3][1][0] = v1[it][iz][iy][ix].v[1][1][0] - CKs * v2L.v[1][1][0];
	  v2[it][iz][iy][ix].v[3][1][1] = v1[it][iz][iy][ix].v[1][1][1] - CKs * v2L.v[1][1][1];
          v2[it][iz][iy][ix].v[3][2][0] = v1[it][iz][iy][ix].v[1][2][0] - CKs * v2L.v[1][2][0];
          v2[it][iz][iy][ix].v[3][2][1] = v1[it][iz][iy][ix].v[1][2][1] - CKs * v2L.v[1][2][1];
	  
	  v2[it][iz][iy][ix].v[0][0][0] = v1[it][iz][iy][ix].v[2][0][0] - CKs * v2L.v[2][0][0];
          v2[it][iz][iy][ix].v[0][0][1] = v1[it][iz][iy][ix].v[2][0][1] - CKs * v2L.v[2][0][1];
	  v2[it][iz][iy][ix].v[0][1][0] = v1[it][iz][iy][ix].v[2][1][0] - CKs * v2L.v[2][1][0];
	  v2[it][iz][iy][ix].v[0][1][1] = v1[it][iz][iy][ix].v[2][1][1] - CKs * v2L.v[2][1][1];
          v2[it][iz][iy][ix].v[0][2][0] = v1[it][iz][iy][ix].v[2][2][0] - CKs * v2L.v[2][2][0];
          v2[it][iz][iy][ix].v[0][2][1] = v1[it][iz][iy][ix].v[2][2][1] - CKs * v2L.v[2][2][1];
	  
	  v2[it][iz][iy][ix].v[1][0][0] = v1[it][iz][iy][ix].v[3][0][0] - CKs * v2L.v[3][0][0];
          v2[it][iz][iy][ix].v[1][0][1] = v1[it][iz][iy][ix].v[3][0][1] - CKs * v2L.v[3][0][1];
	  v2[it][iz][iy][ix].v[1][1][0] = v1[it][iz][iy][ix].v[3][1][0] - CKs * v2L.v[3][1][0];
	  v2[it][iz][iy][ix].v[1][1][1] = v1[it][iz][iy][ix].v[3][1][1] - CKs * v2L.v[3][1][1];
          v2[it][iz][iy][ix].v[1][2][0] = v1[it][iz][iy][ix].v[3][2][0] - CKs * v2L.v[3][2][0];
          v2[it][iz][iy][ix].v[1][2][1] = v1[it][iz][iy][ix].v[3][2][1] - CKs * v2L.v[3][2][1];
	}
      }
    }
  }
}

static void opr_DdagD_alt(QCDSpinor_t v[NT][NZ][NY][NX], QCDMatrix_t u[4][NT][NZ][NY][NX], 
			  QCDSpinor_t w[NT][NZ][NY][NX])
{
  static QCDSpinor_t vt[NT][NZ][NY][NX];
#pragma acc enter data pcreate(vt)

#pragma xmp align v[i][j][*][*] with t(j,i): v, w, v2
#pragma xmp align u[*][i][j][*][*] with t(j,i) shadow
#pragma xmp static_desc :: v, u, w, vt

#pragma xmp reflect (w) width(/periodic/1:1,/periodic/1:1,0,0) acc
#pragma xmp reflect (u) width(0,/periodic/1:1,/periodic/1:1,0,0) acc

  opr_H_alt(vt, u, w);

#pragma xmp reflect (vt) width(/periodic/1:1,/periodic/1:1,0,0) acc
  opr_H_alt(v, u, vt);
}

static void solve_CG_init(real_t *restrict rrp, real_t *restrict rr, QCDMatrix_t u[4][NT][NZ][NY][NX], 
			  QCDSpinor_t x[NT][NZ][NY][NX], QCDSpinor_t r[NT][NZ][NY][NX], 
			  QCDSpinor_t s[NT][NZ][NY][NX], QCDSpinor_t p[NT][NZ][NY][NX])
{
#pragma xmp align u[*][i][j][*][*] with t(j,i) shaow
#pragma xmp align x[i][j][*][*] with t(j,i): x r s p shadow
#pragma xmp static_desc :: u, x, r, s, p
  copy(r, s);
  copy(x, s);
  opr_DdagD_alt(s, u, x);
  axpy(r, -1.0, s);
  copy(p, r);
  *rrp = *rr = norm2(r);
}

static void solve_CG_step(real_t *restrict rrp2, real_t *restrict rr2, QCDMatrix_t u[4][NT][NZ][NY][NX], 
			  QCDSpinor_t x[NT][NZ][NY][NX], QCDSpinor_t r[NT][NZ][NY][NX], 
			  QCDSpinor_t p[NT][NZ][NY][NX], QCDSpinor_t v[NT][NZ][NY][NX])
{
#pragma xmp align u[*][i][j][*][*] with t(j,i) shadow
#pragma xmp align x[i][j][*][*] with t(j,i) x r p v
#pragma xmp static_desc :: u, x, r, p, v
  real_t rrp = *rrp2;

  opr_DdagD_alt(v, u, p);

  real_t pap = dot(v, p);
  real_t cr = rrp/pap;

  axpy(x, cr, p);
  axpy(r, -cr, v);

  real_t rr = norm2(r);
  real_t bk = rr/rrp;

  scal(p, bk);
  axpy(p, 1.0, r);

  *rr2 = *rrp2 = rr;
}

static void solve_CG(const real_t enorm, int *restrict nconv, real_t *restrict diff, QCDSpinor_t xq[NT][NZ][NY][NX], 
		     QCDMatrix_t u[4][NT][NZ][NY][NX], QCDSpinor_t b[NT][NZ][NY][NX])
{
  int niter = 1000;
#pragma xmp align u[*][i][j][*][*] with t(j,i) shadow

  static QCDSpinor_t x[NT][NZ][NY][NX], s[NT][NZ][NY][NX], r[NT][NZ][NY][NX], p[NT][NZ][NY][NX];
#pragma xmp align x[i][j][*][*] with t(j,i) s r p b shadow
#pragma xmp static_desc :: u, xq, b, x, s, r, p

#pragma acc enter data pcreate(x, s, r, p)

  copy(s, b);
  real_t sr = norm2(s);
  real_t snorm = 1.0/sr;
  real_t rr, rrp;
  *nconv = -1;

  solve_CG_init(&rrp, &rr, u, x, r, s, p);

  for(int iter = 0; iter < niter; iter++){
    solve_CG_step(&rrp, &rr, u, x, r, p, s);
    
    if(rr*snorm < enorm){
      *nconv = iter;
      break;
    }
  }

  if(*nconv == -1){
    printf(" not converged\n");
    exit(1);
  }

  copy(xq, x);

  opr_DdagD_alt(r, u, x);
  axpy(r, -1.0, b);
  *diff = norm2(r);
}

static double dtime()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return ((double)(tv.tv_sec) + (double)(tv.tv_usec) * 1.0e-6);
}

static void uinit(QCDMatrix_t u[4][NT][NZ][NY][NX])
{
#pragma xmp align u[*][i][j][*][*] with t(j,i) shad
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
          exit(0);
        }
      }
    }
  }

  fclose(fp);
  int idir, it, iz, iy, ix;
  for(idir = 0; idir < 4; idir++){
#pragma xmp loop (it, iz) on t(iz, it)
    for(it = 0; it < NT; it++){
      for(iz = 0; iz < NZ; iz++){
        for(iy = 0; iy < NY; iy++){
          for(ix = 0; ix < NX; ix++){
            int ix2 = ix % Nx2;
            int iy2 = iy % Ny2;
            int iz2 = iz % Nz2;
            int it2 = it % Nt2;
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

static void setconst(QCDSpinor_t v[NT][NZ][NY][NX], const real_t a)
{
#pragma xmp align v[i][j][*][*] with t(j,i) shadow
  int it, iz, iy, ix, ii, jj, kk;

#pragma xmp loop (it, iz) on t(iz, it)
#pragma acc parallel loop collapse(7) present (v)
  for(it = 0; it < NT; it++)
    for(iz = 0; iz < NZ; iz++)
      for(iy = 0; iy < NY; iy++)
        for(ix = 0; ix < NX; ix++)
          for(ii = 0; ii < ND; ii++)
            for(jj = 0; jj < NCOL; jj++)
              for(kk = 0; kk < 2; kk++)
                v[it][iz][iy][ix].v[ii][jj][kk] = a;
}

static void set_src(const int ic, const int id, const int ix, const int iy, const int iz, const int it,
		    QCDSpinor_t v[NT][NZ][NY][NX])
{
#pragma xmp align v[i][j][*][*] with t(j,i) shadpow
  setconst(v, 0.0);

#pragma xmp task on t(iz, it)
#pragma acc parallel present (v)
  v[it][iz][iy][ix].v[id][ic][0] = 1.0;
}

static void test_mult(QCDMatrix_t u[4][NT][NZ][NY][NX])
{
#pragma xmp align u[*][i][j][*][*] with t(j,i) shadow
  int nrepeat = 100;
  QCDSpinor_t bq2[NT][NZ][NY][NX], xq2[NT][NZ][NY][NX];
#pragma xmp align bq2[i][j][*][*] with t(j,i) xq2 shadow
#pragma acc enter data create(bq2, xq2)
  set_src(0, 0, 0, 0, 0, 0, bq2);

#pragma xmp barrier
  double time0 = dtime();
  for(int i = 0; i < nrepeat; i++){
    opr_DdagD_alt(xq2, u, bq2);
    opr_DdagD_alt(bq2, u, xq2);
  }
#pragma xmp barrier
  double time_tot  = dtime() - time0;
  double fop_mult1 = 2.0 * 1392.0 * (double)(NST);
  double fop_mult  = (double)nrepeat * 2.0 * fop_mult1;

#pragma xmp task on p(1,1)
  {
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

  //  int  namelen, me=xmp_node_num()-1, nprocs;
  //  char processor_name[MPI_MAX_PROCESSOR_NAME];

  //  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  //  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  //  MPI_Get_processor_name(processor_name, &namelen);
  //  printf("Process %d of %d is on %s\n", me, nprocs, processor_name);

  acc_set_device_num((xmp_node_num()-1)%NGPUS+1, acc_device_nvidia);

#pragma xmp task on p(1,1)
  {
    printf("Simple Wilson solver\n\n");
    printf("NX = %3d, NY = %3d, NZ = %3d, NT = %3d\n", NX, NY, NZ, NT);
    //    printf("LX = %3d, LY = %3d, LZ = %3d, LT = %3d\n", NX, NY, NZ/PZ, NT/PT);
    //    printf("(PT x PZ) = (%d x %d)\n", PT, PZ);
    printf("CKs = %10.6f\n", CKs);
    printf("enorm = %12.4e\n", enorm);
  }

#pragma xmp loop on t(*,it)
  for(int it = 0; it < NT; it++)
    corr[it] = 0.0;
  
#pragma acc enter data create(xq, bq)

  uinit(u);
#pragma acc enter data copyin(u)
  test_mult(u);

#pragma xmp task on p(1,1)
  {
    printf("Solver:\n");
    printf("  ic  id   nconv      diff\n");
  }
  double time_tot = 0.0;
  double fop_tot  = 0.0;

  for(int ic = 0; ic < NCOL; ic++){
    for(int id = 0; id < ND; id++){
      set_src(ic, id, 0, 0, 0, 0, bq);
#pragma xmp barrier
      double time0 = dtime();
      solve_CG(enorm, &nconv, &diff, xq, u, bq);
#pragma xmp barrier
      double time1 = dtime();
      time_tot += time1 - time0;
      
#pragma xmp task on p(1,1)
	printf(" %3d %3d  %6d %12.4e\n", ic, id, nconv, diff);
      
      double fop_mult1 = 2.0 * 1392.0 * (double)(NST);
      double fop_mult  = (double)(nconv+2) * fop_mult1;
      double fop_lin   = (double)(4+(nconv+1)*11) * (double)(NVST);
      fop_tot  += fop_lin + fop_mult;

      norm2_t(corr, xq);
    }
  }

#pragma xmp reduction (+:corr) on p(:,*)

#pragma xmp task on p(1,1)
  {
    printf("\nperformance of solver:\n");
    printf("  elapsed time for solver   = %f\n", time_tot);
    printf("  floating point operations = %f\n", fop_tot);
    printf("  performance of solver = %f GFlops\n", fop_tot/time_tot * 1.0e-9);
    printf("\nsolution squared at each time slice:\n");
  }

#pragma xmp task on p(1,*)
#pragma xmp loop on t(*,it)
  for(int it = 0; it < NT; it++)
    printf(" %6d   %16.8e\n", it, corr[it]);

  return 0;
}
