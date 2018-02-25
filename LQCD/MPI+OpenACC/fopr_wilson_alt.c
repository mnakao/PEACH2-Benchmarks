#include "lattice.h"
extern int left, right, up, down;
extern MPI_Comm comm_ud, comm_lr;
extern QCDSpinor_t vt[LT2][LZ2][NY][NX];
extern QCDSpinor_t tmp_QCDSpinor_s[2][LT][NY][NX], tmp_QCDSpinor_r[2][LT][NY][NX];
extern QCDMatrix_t tmp_QCDMatrix_s[4][LT][NY][NX], tmp_QCDMatrix_r[4][LT][NY][NX];
#pragma acc declare present(vt, tmp_QCDSpinor_s, tmp_QCDSpinor_r, tmp_QCDMatrix_s, tmp_QCDMatrix_r)
extern MPI_Request req_u[8], req_mat[2], req_w[4][4], req_spr[4], req_vt[4];

void create_cart(const int pt, const int pz, const int me)
{
  int lr_key = me / pz;
  right = (lr_key != pt-1)? lr_key + 1 : 0;
  left  = (lr_key != 0)?    lr_key - 1 : pt - 1;

  int ud_key = me % pz;
  down   = (ud_key != pz-1)? ud_key + 1 : 0;
  up     = (ud_key != 0)?    ud_key - 1 : pz - 1;
}

static void pack_QCDMatrix(QCDMatrix_t tmp_QCDMatrix_s[4][LT][NY][NX], const QCDMatrix_t u[4][LT2][LZ2][NY][NX])
{
#pragma acc parallel loop collapse(7) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(tmp_QCDMatrix_s[0:4][:][:][:], u[0:4][:][:][:][:])
  for(int ii=0;ii<4;ii++)
    for(int it=0;it<LT2-2;it++)
      for(int iy=0;iy<NY;iy++)
	for(int ix=0;ix<NX;ix++)
	  for(int i=0;i<NCOL;i++)
	    for(int j=0;j<NCOL;j++)
	      for(int k=0;k<2;k++)
		tmp_QCDMatrix_s[ii][it][iy][ix].v[i][j][k] = u[ii][it+1][LZ2-2][iy][ix].v[i][j][k];
}

static void unpack_QCDMatrix(QCDMatrix_t u[4][LT2][LZ2][NY][NX], const QCDMatrix_t tmp_QCDMatrix_r[4][LT][NY][NX])
{
#pragma acc parallel loop collapse(7) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(u[0:4][:][:][:][:], tmp_QCDMatrix_r[0:4][:][:][:])
  for(int ii=0;ii<4;ii++)
    for(int it=0;it<LT2-2;it++)
      for(int iy=0;iy<NY;iy++)
	for(int ix=0;ix<NX;ix++)
	  for(int i=0;i<NCOL;i++)
	    for(int j=0;j<NCOL;j++)
	      for(int k=0;k<2;k++)
		u[ii][it+1][0][iy][ix].v[i][j][k] = tmp_QCDMatrix_r[ii][it][iy][ix].v[i][j][k];
}

static void pack_QCDSpinor(QCDSpinor_t tmp_QCDSpinor_s[2][LT][NY][NX], const QCDSpinor_t w[LT2][LZ2][NY][NX], const int ii, const int jj)
{
#pragma acc parallel loop collapse(6) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(tmp_QCDSpinor_s[0:2][:][:][:], w[0:LT2][:][:][:])
  for(int it=0;it<LT2-2;it++)
    for(int iy=0;iy<NY;iy++)
      for(int ix=0;ix<NX;ix++)
	for(int i=0;i<ND;i++)
	  for(int j=0;j<NCOL;j++)
	    for(int k=0;k<2;k++)
	      tmp_QCDSpinor_s[ii][it][iy][ix].v[i][j][k] = w[it+1][jj][iy][ix].v[i][j][k];
}

static void unpack_QCDSpinor(QCDSpinor_t w[LT2][LZ2][NY][NX], const QCDSpinor_t tmp_QCDSpinor_r[2][LT][NY][NX], const int ii, const int jj)
{
#pragma acc parallel loop collapse(6) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(w[0:LT2][:][:][:], tmp_QCDSpinor_r[0:2][:][:][:])
  for(int it=0;it<LT2-2;it++)
    for(int iy=0;iy<NY;iy++)
      for(int ix=0;ix<NX;ix++)
	for(int i=0;i<ND;i++)
	  for(int j=0;j<NCOL;j++)
	    for(int k=0;k<2;k++)
	      w[it+1][ii][iy][ix].v[i][j][k] = tmp_QCDSpinor_r[jj][it][iy][ix].v[i][j][k];
}

static void opr_H_alt(QCDSpinor_t v2[LT2][LZ2][NY][NX], QCDMatrix_t u[4][LT2][LZ2][NY][NX],
		      QCDSpinor_t v1[LT2][LZ2][NY][NX])
{
  static QCDSpinor_t v2L;
  int it, iz, iy, ix;
#pragma acc enter data pcreate(v2L)

#pragma acc parallel loop collapse(4) private(v2L) vector_length(VECTOR_LENGTH) num_gangs(NUM_GANGS) present(v2[0:LT2][:][:][:], u[0:4][:][:][:][:], v1[0:LT2][:][:][:])
  for(it = 1; it < LT2-1; it++){
    for(iz = 1; iz < LZ2-1; iz++){
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
	  nn = iz + 1; // (iz + 1) % NZ;

	  vt1_0 = v1[it][nn][iy][ix].v[0][0][0] - v1[it][nn][iy][ix].v[2][0][1];
	  vt1_1 = v1[it][nn][iy][ix].v[0][0][1] + v1[it][nn][iy][ix].v[2][0][0];
	  vt1_2 = v1[it][nn][iy][ix].v[0][1][0] - v1[it][nn][iy][ix].v[2][1][1];
	  vt1_3 = v1[it][nn][iy][ix].v[0][1][1] + v1[it][nn][iy][ix].v[2][1][0];
	  vt1_4 = v1[it][nn][iy][ix].v[0][2][0] - v1[it][nn][iy][ix].v[2][2][1];
	  vt1_5 = v1[it][nn][iy][ix].v[0][2][1] + v1[it][nn][iy][ix].v[2][2][0];
    
	  vt2_0 = v1[it][nn][iy][ix].v[1][0][0] + v1[it][nn][iy][ix].v[3][0][1];
	  vt2_1 = v1[it][nn][iy][ix].v[1][0][1] - v1[it][nn][iy][ix].v[3][0][0];
	  vt2_2 = v1[it][nn][iy][ix].v[1][1][0] + v1[it][nn][iy][ix].v[3][1][1];
	  vt2_3 = v1[it][nn][iy][ix].v[1][1][1] - v1[it][nn][iy][ix].v[3][1][0];
	  vt2_4 = v1[it][nn][iy][ix].v[1][2][0] + v1[it][nn][iy][ix].v[3][2][1];
	  vt2_5 = v1[it][nn][iy][ix].v[1][2][1] - v1[it][nn][iy][ix].v[3][2][0];
    
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
	  nn = iz - 1; // (iz + NZ - 1) % NZ;

	  vt1_0 = v1[it][nn][iy][ix].v[0][0][0] + v1[it][nn][iy][ix].v[2][0][1];
	  vt1_1 = v1[it][nn][iy][ix].v[0][0][1] - v1[it][nn][iy][ix].v[2][0][0];
	  vt1_2 = v1[it][nn][iy][ix].v[0][1][0] + v1[it][nn][iy][ix].v[2][1][1];
	  vt1_3 = v1[it][nn][iy][ix].v[0][1][1] - v1[it][nn][iy][ix].v[2][1][0];
	  vt1_4 = v1[it][nn][iy][ix].v[0][2][0] + v1[it][nn][iy][ix].v[2][2][1];
	  vt1_5 = v1[it][nn][iy][ix].v[0][2][1] - v1[it][nn][iy][ix].v[2][2][0];
    
	  vt2_0 = v1[it][nn][iy][ix].v[1][0][0] - v1[it][nn][iy][ix].v[3][0][1];
	  vt2_1 = v1[it][nn][iy][ix].v[1][0][1] + v1[it][nn][iy][ix].v[3][0][0];
	  vt2_2 = v1[it][nn][iy][ix].v[1][1][0] - v1[it][nn][iy][ix].v[3][1][1];
	  vt2_3 = v1[it][nn][iy][ix].v[1][1][1] + v1[it][nn][iy][ix].v[3][1][0];
	  vt2_4 = v1[it][nn][iy][ix].v[1][2][0] - v1[it][nn][iy][ix].v[3][2][1];
	  vt2_5 = v1[it][nn][iy][ix].v[1][2][1] + v1[it][nn][iy][ix].v[3][2][0];
    
	  u_0 = u[idir][it][nn][iy][ix].v[0][0][0];
	  u_1 = u[idir][it][nn][iy][ix].v[0][0][1];
	  u_2 = u[idir][it][nn][iy][ix].v[1][0][0];
	  u_3 = u[idir][it][nn][iy][ix].v[1][0][1];
	  u_4 = u[idir][it][nn][iy][ix].v[2][0][0];
	  u_5 = u[idir][it][nn][iy][ix].v[2][0][1];
    
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
    
	  u_6 = u[idir][it][nn][iy][ix].v[0][1][0];
	  u_7 = u[idir][it][nn][iy][ix].v[0][1][1];
	  u_8 = u[idir][it][nn][iy][ix].v[1][1][0];
	  u_9 = u[idir][it][nn][iy][ix].v[1][1][1];
	  u10 = u[idir][it][nn][iy][ix].v[2][1][0];
	  u11 = u[idir][it][nn][iy][ix].v[2][1][1];
    
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
	  nn = it + 1; //(it + 1) % NT;

	  vt1_0 = 2.0 * v1[nn][iz][iy][ix].v[2][0][0];
	  vt1_1 = 2.0 * v1[nn][iz][iy][ix].v[2][0][1];
	  vt1_2 = 2.0 * v1[nn][iz][iy][ix].v[2][1][0];
	  vt1_3 = 2.0 * v1[nn][iz][iy][ix].v[2][1][1];
	  vt1_4 = 2.0 * v1[nn][iz][iy][ix].v[2][2][0];
	  vt1_5 = 2.0 * v1[nn][iz][iy][ix].v[2][2][1];

	  vt2_0 = 2.0 * v1[nn][iz][iy][ix].v[3][0][0];
	  vt2_1 = 2.0 * v1[nn][iz][iy][ix].v[3][0][1];
	  vt2_2 = 2.0 * v1[nn][iz][iy][ix].v[3][1][0];
	  vt2_3 = 2.0 * v1[nn][iz][iy][ix].v[3][1][1];
	  vt2_4 = 2.0 * v1[nn][iz][iy][ix].v[3][2][0];
	  vt2_5 = 2.0 * v1[nn][iz][iy][ix].v[3][2][1];

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
	  nn = it - 1; //(it + NT - 1) % NT;

	  vt1_0 = 2.0 * v1[nn][iz][iy][ix].v[0][0][0];
	  vt1_1 = 2.0 * v1[nn][iz][iy][ix].v[0][0][1];
	  vt1_2 = 2.0 * v1[nn][iz][iy][ix].v[0][1][0];
	  vt1_3 = 2.0 * v1[nn][iz][iy][ix].v[0][1][1];
	  vt1_4 = 2.0 * v1[nn][iz][iy][ix].v[0][2][0];
	  vt1_5 = 2.0 * v1[nn][iz][iy][ix].v[0][2][1];

	  vt2_0 = 2.0 * v1[nn][iz][iy][ix].v[1][0][0];
	  vt2_1 = 2.0 * v1[nn][iz][iy][ix].v[1][0][1];
	  vt2_2 = 2.0 * v1[nn][iz][iy][ix].v[1][1][0];
	  vt2_3 = 2.0 * v1[nn][iz][iy][ix].v[1][1][1];
	  vt2_4 = 2.0 * v1[nn][iz][iy][ix].v[1][2][0];
	  vt2_5 = 2.0 * v1[nn][iz][iy][ix].v[1][2][1];
    
	  u_0 = u[idir][nn][iz][iy][ix].v[0][0][0];
	  u_1 = u[idir][nn][iz][iy][ix].v[0][0][1];
	  u_2 = u[idir][nn][iz][iy][ix].v[1][0][0];
	  u_3 = u[idir][nn][iz][iy][ix].v[1][0][1];
	  u_4 = u[idir][nn][iz][iy][ix].v[2][0][0];
	  u_5 = u[idir][nn][iz][iy][ix].v[2][0][1];
    
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
    
	  u_6 = u[idir][nn][iz][iy][ix].v[0][1][0];
	  u_7 = u[idir][nn][iz][iy][ix].v[0][1][1];
	  u_8 = u[idir][nn][iz][iy][ix].v[1][1][0];
	  u_9 = u[idir][nn][iz][iy][ix].v[1][1][1];
	  u10 = u[idir][nn][iz][iy][ix].v[2][1][0];
	  u11 = u[idir][nn][iz][iy][ix].v[2][1][1];
    
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

void opr_DdagD_alt(QCDSpinor_t v[LT2][LZ2][NY][NX], QCDMatrix_t u[4][LT2][LZ2][NY][NX], 
		   QCDSpinor_t w[LT2][LZ2][NY][NX], const int n)
{
  //  static QCDSpinor_t vt[LT2][LZ2][NY][NX];
  //#pragma acc enter data pcreate(vt)


  MPI_Status st[10];
  MPI_Request req[10];

  int QCDSpinor_zyxvec = LZ * yx_Spinor;
  int QCDMatrix_zyxvec = LZ * yx_Matrix;
  int QCDSpinor_tyxvec = LT * yx_Spinor;
  int QCDMatrix_tyxvec = 4*LT*yx_Matrix;

  //  static QCDSpinor_t tmp_QCDSpinor_s[2][LT][NY][NX], tmp_QCDSpinor_r[2][LT][NY][NX];
  //  static QCDMatrix_t tmp_QCDMatrix_s[4][LT][NY][NX], tmp_QCDMatrix_r[4][LT][NY][NX];
  //#pragma acc enter data pcreate(tmp_QCDSpinor_s, tmp_QCDSpinor_r, tmp_QCDMatrix_s, tmp_QCDMatrix_r)

#ifdef _PROF
  double tmp = dtime();
#endif
  pack_QCDMatrix(tmp_QCDMatrix_s, u);
#ifdef _PROF
  prof_t[PACK] += dtime() - tmp;
  tmp = dtime();
#endif

  /*
#pragma acc data present(u[0:4][:][:][:][:], tmp_QCDMatrix_r, tmp_QCDMatrix_s)
  {
#pragma acc host_data use_device(u, tmp_QCDMatrix_r, tmp_QCDMatrix_s)
    {
      for(int i=0;i<4;i++){
        MPI_Irecv(&u[i][0][1][0][0],     QCDMatrix_zyxvec, MPI_DOUBLE, left,  i, comm_lr, req+0+i*2);
        MPI_Isend(&u[i][LT2-2][1][0][0], QCDMatrix_zyxvec, MPI_DOUBLE, right, i, comm_lr, req+1+i*2);
      }
      MPI_Irecv(tmp_QCDMatrix_r, QCDMatrix_tyxvec, MPI_DOUBLE, up,   5, comm_ud, req+8);
      MPI_Isend(tmp_QCDMatrix_s, QCDMatrix_tyxvec, MPI_DOUBLE, down, 5, comm_ud, req+9);
    }
  }

  MPI_Waitall(10, req, st);
  */

  MPI_Startall(8, req_u);
  MPI_Startall(2, req_mat);
  MPI_Waitall(8, req_u, MPI_STATUSES_IGNORE);
  MPI_Waitall(2, req_mat, MPI_STATUSES_IGNORE);
#ifdef _PROF
  prof_t[COMM] += dtime() - tmp;
  tmp = dtime();
#endif
  unpack_QCDMatrix(u, tmp_QCDMatrix_r);
  pack_QCDSpinor(tmp_QCDSpinor_s, w, 0, 1);
  pack_QCDSpinor(tmp_QCDSpinor_s, w, 1, LZ2-2);
#ifdef _PROF
  prof_t[PACK] += dtime() - tmp;
  tmp = dtime();
#endif
  
  /*
#pragma acc data present(w[0:LT2][:][:][:], tmp_QCDSpinor_r, tmp_QCDSpinor_s)
  {
#pragma acc host_data use_device(w, tmp_QCDSpinor_r, tmp_QCDSpinor_s)
    {
      MPI_Irecv(&w[LT2-1][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, req+0);
      MPI_Irecv(&w[0][1][0][0],     QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, req+1);
      MPI_Isend(&w[1][1][0][0],     QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, req+2);
      MPI_Isend(&w[LT2-2][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, req+3);

      MPI_Irecv(&tmp_QCDSpinor_r[1][0][0][0], QCDSpinor_tyxvec, MPI_DOUBLE, down, 12, comm_ud, req+4);
      MPI_Irecv(&tmp_QCDSpinor_r[0][0][0][0], QCDSpinor_tyxvec, MPI_DOUBLE, up,   13, comm_ud, req+5);
      MPI_Isend(&tmp_QCDSpinor_s[0][0][0][0], QCDSpinor_tyxvec, MPI_DOUBLE, up,   12, comm_ud, req+6);
      MPI_Isend(&tmp_QCDSpinor_s[1][0][0][0], QCDSpinor_tyxvec, MPI_DOUBLE, down, 13, comm_ud, req+7);
    }
  }

  MPI_Waitall(8, req, st);
  */
  MPI_Startall(4, req_w[n]);
  MPI_Startall(4, req_spr);
  MPI_Waitall(4, req_w[n], MPI_STATUSES_IGNORE);
  MPI_Waitall(4, req_spr, MPI_STATUSES_IGNORE);
#ifdef _PROF
  prof_t[COMM] += dtime() - tmp;
  tmp = dtime();
#endif
  unpack_QCDSpinor(w, tmp_QCDSpinor_r, 0, 0);
  unpack_QCDSpinor(w, tmp_QCDSpinor_r, LZ2-1, 1);
#ifdef _PROF
  prof_t[PACK] += dtime() - tmp;
  tmp = dtime();
#endif
  opr_H_alt(vt, u, w);
#ifdef _PROF
  prof_t[OPR] += dtime() - tmp;
  tmp = dtime();
#endif
  pack_QCDSpinor(tmp_QCDSpinor_s, vt, 0, 1);
  pack_QCDSpinor(tmp_QCDSpinor_s, vt, 1, LZ2-2);
#ifdef _PROF
  prof_t[PACK] += dtime() - tmp;
  tmp = dtime();
#endif
  
  /*
#pragma acc data present (tmp_QCDSpinor_r, tmp_QCDSpinor_s, vt[0:LT2][:][:][:])
  {
#pragma acc host_data use_device(vt, tmp_QCDSpinor_r, tmp_QCDSpinor_s)
    {
      MPI_Irecv(&vt[LT2-1][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, req+0);
      MPI_Irecv(&vt[0][1][0][0],     QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, req+1);
      MPI_Isend(&vt[1][1][0][0],     QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, req+2);
      MPI_Isend(&vt[LT2-2][1][0][0], QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, req+3);

      MPI_Irecv(&tmp_QCDSpinor_r[1][0][0][0], QCDSpinor_tyxvec, MPI_DOUBLE, down, 12, comm_ud, req+4);
      MPI_Irecv(&tmp_QCDSpinor_r[0][0][0][0], QCDSpinor_tyxvec, MPI_DOUBLE, up,   13, comm_ud, req+5);
      MPI_Isend(&tmp_QCDSpinor_s[0][0][0][0], QCDSpinor_tyxvec, MPI_DOUBLE, up,   12, comm_ud, req+6);
      MPI_Isend(&tmp_QCDSpinor_s[1][0][0][0], QCDSpinor_tyxvec, MPI_DOUBLE, down, 13, comm_ud, req+7);
    }
  }

  MPI_Waitall(8, req, st);
  */
  MPI_Startall(4, req_vt);
  MPI_Startall(4, req_spr);
  MPI_Waitall(4, req_vt, MPI_STATUSES_IGNORE);
  MPI_Waitall(4, req_spr, MPI_STATUSES_IGNORE);
#ifdef _PROF
  prof_t[COMM] += dtime() - tmp;
  tmp = dtime();
#endif
  unpack_QCDSpinor(vt, tmp_QCDSpinor_r, 0, 0);
  unpack_QCDSpinor(vt, tmp_QCDSpinor_r, LZ2-1, 1);
#ifdef _PROF
  prof_t[PACK] += dtime() - tmp;
  tmp = dtime();
#endif
  opr_H_alt(v, u, vt);
#ifdef _PROF
  prof_t[OPR] += dtime() - tmp;
#endif
}
