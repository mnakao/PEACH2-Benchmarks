#include "lattice.h"
static int left, right, up, down;
extern MPI_Comm comm_ud, comm_lr;

void create_cart(const int pt, const int pz, const int me)
{
  int lr_key = me / pz;
  right = (lr_key != pt-1)? lr_key + 1 : 0;
  left  = (lr_key != 0)?    lr_key - 1 : pt - 1;

  int ud_key = me % pz;
  down   = (ud_key != pz-1)? ud_key + 1 : 0;
  up     = (ud_key != 0)?    ud_key - 1 : pz - 1;
}

__global__ static void pack_QCDMatrix(real_t* __restrict__ v, const real_t* __restrict__ w)
{
  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);
  int tyx_Matrix = yx_Matrix*LT;
  while(ivx < 4*tyx_Matrix){
    int hi = ivx / tyx_Matrix;
    int mi = (ivx - hi*tyx_Matrix) / yx_Matrix;
    int lo = ivx - hi*tyx_Matrix - mi*yx_Matrix;
    int rhs = hi*LT2*LZ2*yx_Matrix + (mi+1)*LZ2*yx_Matrix + (LZ2-2)*yx_Matrix + lo;
    v[ivx] = w[rhs];
    ivx += blockDim.x * gridDim.x;
  }
}

__global__ static void unpack_QCDMatrix(real_t* __restrict__ v, const real_t* __restrict__ w)
{
  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);
  int tyx_Matrix = yx_Matrix*LT;
  while(ivx < 4*tyx_Matrix){
    int hi = ivx / tyx_Matrix;
    int mi = (ivx - hi*tyx_Matrix)/yx_Matrix;
    int lo = ivx - hi*tyx_Matrix - mi*yx_Matrix;
    int lhs = hi*LT2*LZ2*yx_Matrix + (mi+1)*LZ2*yx_Matrix + lo;
    v[lhs] = w[ivx];
    ivx += blockDim.x * gridDim.x;
  }
}

__global__ static void pack_QCDSpinor(real_t* __restrict__ v, const real_t* __restrict__ w, const int v1, const int v2)
{
  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);
  while(ivx < LT*yx_Spinor){
    int hi = ivx / yx_Spinor;
    int lo = ivx - hi * yx_Spinor;
    int lhs = v1*LT*yx_Spinor + ivx;
    int rhs = (hi+1)*LZ2*yx_Spinor + v2*yx_Spinor + lo;
    v[lhs] = w[rhs];
    ivx += blockDim.x * gridDim.x;
  }
}

__global__ static void unpack_QCDSpinor(real_t* __restrict__ v, const real_t* __restrict__ w, const int v1, const int v2)
{
  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);
  while(ivx < LT*yx_Spinor){
    int hi = ivx / yx_Spinor;
    int lo = ivx - hi * yx_Spinor;
    int lhs = (hi+1)*LZ2*yx_Spinor + v1*yx_Spinor + lo;
    int rhs = v2*LT*yx_Spinor + ivx;
    v[lhs] = w[rhs];
    ivx += blockDim.x * gridDim.x;
  }
}

__global__ static void opr_H_dev(real_t* __restrict__ v2, const real_t* __restrict__ u, 
				 const real_t* __restrict__ v1)
{
  real_t v2L[NVC*ND];
  int ivx = IDXV(threadIdx.x, blockIdx.x, blockDim.x);
  
  while(ivx < (LT2-2)*(LZ2-2)*NY*NX){
    real_t u_0, u_1, u_2, u_3, u_4, u_5;
    real_t u_6, u_7, u_8, u_9, u10, u11;
    real_t u12, u13, u14, u15, u16, u17;
    real_t vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5;
    real_t vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5;
    real_t wt1r, wt1i, wt2r, wt2i;

    int tmp_it  = ivx / ((LZ2-2)*NY*NX);
    int tmp_iz  = (ivx - tmp_it * ((LZ2-2)*NY*NX)) / (NY*NX);
    int tmp_iyx = ivx % (NY*NX);
    int ivx2 = (tmp_it+1)*(LZ2*NY*NX) + (tmp_iz+1)*NY*NX + tmp_iyx;
  
    // mult_xp
    int idir = 0;
    int ic;
    int iyzt = ivx2 / NX;
    int ix   = ivx2 % NX;
    int nn   = (ix + 1) % NX;
    int ivn  = nn + iyzt * NX;
    int ivg  = ivx2 + NST2 * idir;
    
    vt1_0 = v1[IDX(0,0,ivn)] - v1[IDX(1,3,ivn)];
    vt1_1 = v1[IDX(1,0,ivn)] + v1[IDX(0,3,ivn)];
    vt1_2 = v1[IDX(2,0,ivn)] - v1[IDX(3,3,ivn)];
    vt1_3 = v1[IDX(3,0,ivn)] + v1[IDX(2,3,ivn)];
    vt1_4 = v1[IDX(4,0,ivn)] - v1[IDX(5,3,ivn)];
    vt1_5 = v1[IDX(5,0,ivn)] + v1[IDX(4,3,ivn)];
    
    vt2_0 = v1[IDX(0,1,ivn)] - v1[IDX(1,2,ivn)];
    vt2_1 = v1[IDX(1,1,ivn)] + v1[IDX(0,2,ivn)];
    vt2_2 = v1[IDX(2,1,ivn)] - v1[IDX(3,2,ivn)];
    vt2_3 = v1[IDX(3,1,ivn)] + v1[IDX(2,2,ivn)];
    vt2_4 = v1[IDX(4,1,ivn)] - v1[IDX(5,2,ivn)];
    vt2_5 = v1[IDX(5,1,ivn)] + v1[IDX(4,2,ivn)];
   
    ic = 0;
    u_0 = u[IDG(0,ic,ivg)];
    u_1 = u[IDG(1,ic,ivg)];
    u_2 = u[IDG(2,ic,ivg)];
    u_3 = u[IDG(3,ic,ivg)];
    u_4 = u[IDG(4,ic,ivg)];
    u_5 = u[IDG(5,ic,ivg)];
    
    wt1r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
    v2L[    2*(ic + NCOL*0)] =  wt1r;    v2L[1 + 2*(ic + NCOL*0)] =  wt1i;
    v2L[    2*(ic + NCOL*1)] =  wt2r;    v2L[1 + 2*(ic + NCOL*1)] =  wt2i;
    v2L[    2*(ic + NCOL*2)] =  wt2i;    v2L[1 + 2*(ic + NCOL*2)] = -wt2r;
    v2L[    2*(ic + NCOL*3)] =  wt1i;    v2L[1 + 2*(ic + NCOL*3)] = -wt1r;

    ic = 1;
    u_6 = u[IDG(0,ic,ivg)];
    u_7 = u[IDG(1,ic,ivg)];
    u_8 = u[IDG(2,ic,ivg)];
    u_9 = u[IDG(3,ic,ivg)];
    u10 = u[IDG(4,ic,ivg)];
    u11 = u[IDG(5,ic,ivg)];

    wt1r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
  
    v2L[    2*(ic + NCOL*0)] =  wt1r;    v2L[1 + 2*(ic + NCOL*0)] =  wt1i;
    v2L[    2*(ic + NCOL*1)] =  wt2r;    v2L[1 + 2*(ic + NCOL*1)] =  wt2i;
    v2L[    2*(ic + NCOL*2)] =  wt2i;    v2L[1 + 2*(ic + NCOL*2)] = -wt2r;
    v2L[    2*(ic + NCOL*3)] =  wt1i;    v2L[1 + 2*(ic + NCOL*3)] = -wt1r;

    ic = 2;
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

    v2L[    2*(ic + NCOL*0)] =  wt1r;    v2L[1 + 2*(ic + NCOL*0)] =  wt1i;
    v2L[    2*(ic + NCOL*1)] =  wt2r;    v2L[1 + 2*(ic + NCOL*1)] =  wt2i;
    v2L[    2*(ic + NCOL*2)] =  wt2i;    v2L[1 + 2*(ic + NCOL*2)] = -wt2r;
    v2L[    2*(ic + NCOL*3)] =  wt1i;    v2L[1 + 2*(ic + NCOL*3)] = -wt1r;
    
    // mult_xm
    nn  = (ix + NX - 1) % NX;
    ivn = nn + iyzt * NX;
    ivg = ivn + NST2 * idir;

    vt1_0 = v1[IDX(0,0,ivn)] + v1[IDX(1,3,ivn)];
    vt1_1 = v1[IDX(1,0,ivn)] - v1[IDX(0,3,ivn)];
    vt1_2 = v1[IDX(2,0,ivn)] + v1[IDX(3,3,ivn)];
    vt1_3 = v1[IDX(3,0,ivn)] - v1[IDX(2,3,ivn)];
    vt1_4 = v1[IDX(4,0,ivn)] + v1[IDX(5,3,ivn)];
    vt1_5 = v1[IDX(5,0,ivn)] - v1[IDX(4,3,ivn)];
    
    vt2_0 = v1[IDX(0,1,ivn)] + v1[IDX(1,2,ivn)];
    vt2_1 = v1[IDX(1,1,ivn)] - v1[IDX(0,2,ivn)];
    vt2_2 = v1[IDX(2,1,ivn)] + v1[IDX(3,2,ivn)];
    vt2_3 = v1[IDX(3,1,ivn)] - v1[IDX(2,2,ivn)];
    vt2_4 = v1[IDX(4,1,ivn)] + v1[IDX(5,2,ivn)];
    vt2_5 = v1[IDX(5,1,ivn)] - v1[IDX(4,2,ivn)];

    ic = 0;
    u_0 = u[IDG(0,0,ivg)];
    u_1 = u[IDG(1,0,ivg)];
    u_2 = u[IDG(0,1,ivg)];
    u_3 = u[IDG(1,1,ivg)];
    u_4 = u[IDG(0,2,ivg)];
    u_5 = u[IDG(1,2,ivg)];
    
    wt1r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    v2L[  2*(ic+NCOL*2)] += -wt2i;    v2L[1+2*(ic+NCOL*2)] += +wt2r;
    v2L[  2*(ic+NCOL*3)] += -wt1i;    v2L[1+2*(ic+NCOL*3)] += +wt1r;
    
    ic = 1;
    u_6 = u[IDG(2,0,ivg)];
    u_7 = u[IDG(3,0,ivg)];
    u_8 = u[IDG(2,1,ivg)];
    u_9 = u[IDG(3,1,ivg)];
    u10 = u[IDG(2,2,ivg)];
    u11 = u[IDG(3,2,ivg)];
    
    wt1r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    v2L[  2*(ic+NCOL*2)] += -wt2i;    v2L[1+2*(ic+NCOL*2)] += +wt2r;
    v2L[  2*(ic+NCOL*3)] += -wt1i;    v2L[1+2*(ic+NCOL*3)] += +wt1r;

    ic = 2;
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

    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    v2L[  2*(ic+NCOL*2)] += -wt2i;    v2L[1+2*(ic+NCOL*2)] += +wt2r;
    v2L[  2*(ic+NCOL*3)] += -wt1i;    v2L[1+2*(ic+NCOL*3)] += +wt1r;
    
    // mult_yp
    idir = 1;
    int izt = ivx2 / (NX * NY);
    int iy  = (ivx2 / NX) % NY;
    nn = (iy + 1) % NY;
    ivn = ix + nn * NX + izt * NX * NY;
    ivg = ivx2 + NST2 * idir;

    vt1_0 = v1[IDX(0,0,ivn)] + v1[IDX(0,3,ivn)];
    vt1_1 = v1[IDX(1,0,ivn)] + v1[IDX(1,3,ivn)];
    vt1_2 = v1[IDX(2,0,ivn)] + v1[IDX(2,3,ivn)];
    vt1_3 = v1[IDX(3,0,ivn)] + v1[IDX(3,3,ivn)];
    vt1_4 = v1[IDX(4,0,ivn)] + v1[IDX(4,3,ivn)];
    vt1_5 = v1[IDX(5,0,ivn)] + v1[IDX(5,3,ivn)];
    
    vt2_0 = v1[IDX(0,1,ivn)] - v1[IDX(0,2,ivn)];
    vt2_1 = v1[IDX(1,1,ivn)] - v1[IDX(1,2,ivn)];
    vt2_2 = v1[IDX(2,1,ivn)] - v1[IDX(2,2,ivn)];
    vt2_3 = v1[IDX(3,1,ivn)] - v1[IDX(3,2,ivn)];
    vt2_4 = v1[IDX(4,1,ivn)] - v1[IDX(4,2,ivn)];
    vt2_5 = v1[IDX(5,1,ivn)] - v1[IDX(5,2,ivn)];

    ic = 0;
    u_0 = u[IDG(0,ic,ivg)];
    u_1 = u[IDG(1,ic,ivg)];
    u_2 = u[IDG(2,ic,ivg)];
    u_3 = u[IDG(3,ic,ivg)];
    u_4 = u[IDG(4,ic,ivg)];
    u_5 = u[IDG(5,ic,ivg)];

    wt1r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    v2L[  2*(ic+NCOL*2)] += -wt2r;    v2L[1+2*(ic+NCOL*2)] += -wt2i;
    v2L[  2*(ic+NCOL*3)] +=  wt1r;    v2L[1+2*(ic+NCOL*3)] +=  wt1i;

    ic = 1;
    u_6 = u[IDG(0,ic,ivg)];
    u_7 = u[IDG(1,ic,ivg)];
    u_8 = u[IDG(2,ic,ivg)];
    u_9 = u[IDG(3,ic,ivg)];
    u10 = u[IDG(4,ic,ivg)];
    u11 = u[IDG(5,ic,ivg)];

    wt1r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    v2L[  2*(ic+NCOL*2)] += -wt2r;    v2L[1+2*(ic+NCOL*2)] += -wt2i;
    v2L[  2*(ic+NCOL*3)] +=  wt1r;    v2L[1+2*(ic+NCOL*3)] +=  wt1i;

    ic = 2;
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

    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    v2L[  2*(ic+NCOL*2)] += -wt2r;    v2L[1+2*(ic+NCOL*2)] += -wt2i;
    v2L[  2*(ic+NCOL*3)] +=  wt1r;    v2L[1+2*(ic+NCOL*3)] +=  wt1i;
    
    // mult_ym
    nn  = (iy + NY - 1) % NY;
    ivn = ix + nn * NX + izt * NX * NY;
    ivg = ivn + NST2 * idir;

    vt1_0 = v1[IDX(0,0,ivn)] - v1[IDX(0,3,ivn)];
    vt1_1 = v1[IDX(1,0,ivn)] - v1[IDX(1,3,ivn)];
    vt1_2 = v1[IDX(2,0,ivn)] - v1[IDX(2,3,ivn)];
    vt1_3 = v1[IDX(3,0,ivn)] - v1[IDX(3,3,ivn)];
    vt1_4 = v1[IDX(4,0,ivn)] - v1[IDX(4,3,ivn)];
    vt1_5 = v1[IDX(5,0,ivn)] - v1[IDX(5,3,ivn)];
    
    vt2_0 = v1[IDX(0,1,ivn)] + v1[IDX(0,2,ivn)];
    vt2_1 = v1[IDX(1,1,ivn)] + v1[IDX(1,2,ivn)];
    vt2_2 = v1[IDX(2,1,ivn)] + v1[IDX(2,2,ivn)];
    vt2_3 = v1[IDX(3,1,ivn)] + v1[IDX(3,2,ivn)];
    vt2_4 = v1[IDX(4,1,ivn)] + v1[IDX(4,2,ivn)];
    vt2_5 = v1[IDX(5,1,ivn)] + v1[IDX(5,2,ivn)];

    ic = 0;
    u_0 = u[IDG(0,0,ivg)];
    u_1 = u[IDG(1,0,ivg)];
    u_2 = u[IDG(0,1,ivg)];
    u_3 = u[IDG(1,1,ivg)];
    u_4 = u[IDG(0,2,ivg)];
    u_5 = u[IDG(1,2,ivg)];

    wt1r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    v2L[  2*(ic+NCOL*2)] +=  wt2r;    v2L[1+2*(ic+NCOL*2)] +=  wt2i;
    v2L[  2*(ic+NCOL*3)] += -wt1r;    v2L[1+2*(ic+NCOL*3)] += -wt1i;

    ic = 1;
    u_6 = u[IDG(2,0,ivg)];
    u_7 = u[IDG(3,0,ivg)];
    u_8 = u[IDG(2,1,ivg)];
    u_9 = u[IDG(3,1,ivg)];
    u10 = u[IDG(2,2,ivg)];
    u11 = u[IDG(3,2,ivg)];

    wt1r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    v2L[  2*(ic+NCOL*2)] +=  wt2r;    v2L[1+2*(ic+NCOL*2)] +=  wt2i;
    v2L[  2*(ic+NCOL*3)] += -wt1r;    v2L[1+2*(ic+NCOL*3)] += -wt1i;

    ic = 2;
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

    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    v2L[  2*(ic+NCOL*2)] +=  wt2r;    v2L[1+2*(ic+NCOL*2)] +=  wt2i;
    v2L[  2*(ic+NCOL*3)] += -wt1r;    v2L[1+2*(ic+NCOL*3)] += -wt1i;
    
    // mult_zp
    idir = 2;
    int it = ivx2/(NX*NY*LZ2);
    int iz = (ivx2/(NX*NY)) % LZ2;
    int ixy = ivx2 % (NX*NY);
    nn = iz + 1;
    ivn = ixy + nn*NX*NY + it*NX*NY*LZ2;
    ivg = ivx2 + NST2 * idir;

    vt1_0 = v1[IDX(0,0,ivn)] - v1[IDX(1,2,ivn)];
    vt1_1 = v1[IDX(1,0,ivn)] + v1[IDX(0,2,ivn)];
    vt1_2 = v1[IDX(2,0,ivn)] - v1[IDX(3,2,ivn)];
    vt1_3 = v1[IDX(3,0,ivn)] + v1[IDX(2,2,ivn)];
    vt1_4 = v1[IDX(4,0,ivn)] - v1[IDX(5,2,ivn)];
    vt1_5 = v1[IDX(5,0,ivn)] + v1[IDX(4,2,ivn)];
    
    vt2_0 = v1[IDX(0,1,ivn)] + v1[IDX(1,3,ivn)];
    vt2_1 = v1[IDX(1,1,ivn)] - v1[IDX(0,3,ivn)];
    vt2_2 = v1[IDX(2,1,ivn)] + v1[IDX(3,3,ivn)];
    vt2_3 = v1[IDX(3,1,ivn)] - v1[IDX(2,3,ivn)];
    vt2_4 = v1[IDX(4,1,ivn)] + v1[IDX(5,3,ivn)];
    vt2_5 = v1[IDX(5,1,ivn)] - v1[IDX(4,3,ivn)];

    ic = 0;
    u_0 = u[IDG(0,ic,ivg)];
    u_1 = u[IDG(1,ic,ivg)];
    u_2 = u[IDG(2,ic,ivg)];
    u_3 = u[IDG(3,ic,ivg)];
    u_4 = u[IDG(4,ic,ivg)];
    u_5 = u[IDG(5,ic,ivg)];

    wt1r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    v2L[  2*(ic+NCOL*2)] +=  wt1i;    v2L[1+2*(ic+NCOL*2)] += -wt1r;
    v2L[  2*(ic+NCOL*3)] += -wt2i;    v2L[1+2*(ic+NCOL*3)] +=  wt2r;

    ic = 1;
    u_6 = u[IDG(0,ic,ivg)];
    u_7 = u[IDG(1,ic,ivg)];
    u_8 = u[IDG(2,ic,ivg)];
    u_9 = u[IDG(3,ic,ivg)];
    u10 = u[IDG(4,ic,ivg)];
    u11 = u[IDG(5,ic,ivg)];

    wt1r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    v2L[  2*(ic+NCOL*2)] +=  wt1i;    v2L[1+2*(ic+NCOL*2)] += -wt1r;
    v2L[  2*(ic+NCOL*3)] += -wt2i;    v2L[1+2*(ic+NCOL*3)] +=  wt2r;

    ic = 2;
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
    
    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    v2L[  2*(ic+NCOL*2)] +=  wt1i;    v2L[1+2*(ic+NCOL*2)] += -wt1r;
    v2L[  2*(ic+NCOL*3)] += -wt2i;    v2L[1+2*(ic+NCOL*3)] +=  wt2r;

    // mult_zm
    nn = iz - 1;   //    nn = (iz + NZ - 1) % NZ;
    ivn = ixy + nn*NX*NY + it*NX*NY*LZ2;
    ivg = ivn + NST2 * idir;

    vt1_0 = v1[IDX(0,0,ivn)] + v1[IDX(1,2,ivn)];
    vt1_1 = v1[IDX(1,0,ivn)] - v1[IDX(0,2,ivn)];
    vt1_2 = v1[IDX(2,0,ivn)] + v1[IDX(3,2,ivn)];
    vt1_3 = v1[IDX(3,0,ivn)] - v1[IDX(2,2,ivn)];
    vt1_4 = v1[IDX(4,0,ivn)] + v1[IDX(5,2,ivn)];
    vt1_5 = v1[IDX(5,0,ivn)] - v1[IDX(4,2,ivn)];
    
    vt2_0 = v1[IDX(0,1,ivn)] - v1[IDX(1,3,ivn)];
    vt2_1 = v1[IDX(1,1,ivn)] + v1[IDX(0,3,ivn)];
    vt2_2 = v1[IDX(2,1,ivn)] - v1[IDX(3,3,ivn)];
    vt2_3 = v1[IDX(3,1,ivn)] + v1[IDX(2,3,ivn)];
    vt2_4 = v1[IDX(4,1,ivn)] - v1[IDX(5,3,ivn)];
    vt2_5 = v1[IDX(5,1,ivn)] + v1[IDX(4,3,ivn)];

    ic = 0;
    u_0 = u[IDG(0,0,ivg)];
    u_1 = u[IDG(1,0,ivg)];
    u_2 = u[IDG(0,1,ivg)];
    u_3 = u[IDG(1,1,ivg)];
    u_4 = u[IDG(0,2,ivg)];
    u_5 = u[IDG(1,2,ivg)];

    wt1r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    v2L[  2*(ic+NCOL*2)] += -wt1i;    v2L[1+2*(ic+NCOL*2)] +=  wt1r;
    v2L[  2*(ic+NCOL*3)] +=  wt2i;    v2L[1+2*(ic+NCOL*3)] += -wt2r;
    
    ic = 1;
    u_6 = u[IDG(2,0,ivg)];
    u_7 = u[IDG(3,0,ivg)];
    u_8 = u[IDG(2,1,ivg)];
    u_9 = u[IDG(3,1,ivg)];
    u10 = u[IDG(2,2,ivg)];
    u11 = u[IDG(3,2,ivg)];

    wt1r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    v2L[  2*(ic+NCOL*2)] += -wt1i;    v2L[1+2*(ic+NCOL*2)] +=  wt1r;
    v2L[  2*(ic+NCOL*3)] +=  wt2i;    v2L[1+2*(ic+NCOL*3)] += -wt2r;

    ic = 2;
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

    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    v2L[  2*(ic+NCOL*2)] += -wt1i;    v2L[1+2*(ic+NCOL*2)] +=  wt1r;
    v2L[  2*(ic+NCOL*3)] +=  wt2i;    v2L[1+2*(ic+NCOL*3)] += -wt2r;

    // mult_tp
    idir = 3;
    nn = it + 1;
    int ixyz = ivx2 % (NX*NY*LZ2);
    ivn  = ixyz + nn*NX*NY*LZ2;
    ivg  = ivx2 + NST2 * idir;

    vt1_0 = 2.0f * v1[IDX(0,2,ivn)];
    vt1_1 = 2.0f * v1[IDX(1,2,ivn)];
    vt1_2 = 2.0f * v1[IDX(2,2,ivn)];
    vt1_3 = 2.0f * v1[IDX(3,2,ivn)];
    vt1_4 = 2.0f * v1[IDX(4,2,ivn)];
    vt1_5 = 2.0f * v1[IDX(5,2,ivn)];
    
    vt2_0 = 2.0f * v1[IDX(0,3,ivn)];
    vt2_1 = 2.0f * v1[IDX(1,3,ivn)];
    vt2_2 = 2.0f * v1[IDX(2,3,ivn)];
    vt2_3 = 2.0f * v1[IDX(3,3,ivn)];
    vt2_4 = 2.0f * v1[IDX(4,3,ivn)];
    vt2_5 = 2.0f * v1[IDX(5,3,ivn)];
    
    ic = 0;
    u_0 = u[IDG(0,ic,ivg)];
    u_1 = u[IDG(1,ic,ivg)];
    u_2 = u[IDG(2,ic,ivg)];
    u_3 = u[IDG(3,ic,ivg)];
    u_4 = u[IDG(4,ic,ivg)];
    u_5 = u[IDG(5,ic,ivg)];

    wt1r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GXr(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GXi(u_0, u_1, u_2, u_3, u_4, u_5,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

    v2L[  2*(ic+NCOL*2)] +=  wt1r;    v2L[1+2*(ic+NCOL*2)] +=  wt1i;
    v2L[  2*(ic+NCOL*3)] +=  wt2r;    v2L[1+2*(ic+NCOL*3)] +=  wt2i;

    ic = 1;
    u_6 = u[IDG(0,ic,ivg)];
    u_7 = u[IDG(1,ic,ivg)];
    u_8 = u[IDG(2,ic,ivg)];
    u_9 = u[IDG(3,ic,ivg)];
    u10 = u[IDG(4,ic,ivg)];
    u11 = u[IDG(5,ic,ivg)];

    wt1r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
		    vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GXr(u_6, u_7, u_8, u_9, u10, u11,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GXi(u_6, u_7, u_8, u_9, u10, u11,
		    vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    
    v2L[  2*(ic+NCOL*2)] +=  wt1r;    v2L[1+2*(ic+NCOL*2)] +=  wt1i;
    v2L[  2*(ic+NCOL*3)] +=  wt2r;    v2L[1+2*(ic+NCOL*3)] +=  wt2i;
    
    ic = 2;
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

    v2L[  2*(ic+NCOL*2)] +=  wt1r;    v2L[1+2*(ic+NCOL*2)] +=  wt1i;
    v2L[  2*(ic+NCOL*3)] +=  wt2r;    v2L[1+2*(ic+NCOL*3)] +=  wt2i;
    
    // mult_tm
    nn = it - 1;   //    nn  = (it + NT - 1) % NT;
    ivn = ixyz + nn*NX*NY*LZ2;
    ivg = ivn + NST2 * idir;

    vt1_0 = 2.0f * v1[IDX(0,0,ivn)];
    vt1_1 = 2.0f * v1[IDX(1,0,ivn)];
    vt1_2 = 2.0f * v1[IDX(2,0,ivn)];
    vt1_3 = 2.0f * v1[IDX(3,0,ivn)];
    vt1_4 = 2.0f * v1[IDX(4,0,ivn)];
    vt1_5 = 2.0f * v1[IDX(5,0,ivn)];
    
    vt2_0 = 2.0f * v1[IDX(0,1,ivn)];
    vt2_1 = 2.0f * v1[IDX(1,1,ivn)];
    vt2_2 = 2.0f * v1[IDX(2,1,ivn)];
    vt2_3 = 2.0f * v1[IDX(3,1,ivn)];
    vt2_4 = 2.0f * v1[IDX(4,1,ivn)];
    vt2_5 = 2.0f * v1[IDX(5,1,ivn)];

    ic = 0;
    u_0 = u[IDG(0,0,ivg)];
    u_1 = u[IDG(1,0,ivg)];
    u_2 = u[IDG(0,1,ivg)];
    u_3 = u[IDG(1,1,ivg)];
    u_4 = u[IDG(0,2,ivg)];
    u_5 = u[IDG(1,2,ivg)];

    wt1r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GDXr(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GDXi(u_0, u_1, u_2, u_3, u_4, u_5,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;

    ic = 1;
    u_6 = u[IDG(2,0,ivg)];
    u_7 = u[IDG(3,0,ivg)];
    u_8 = u[IDG(2,1,ivg)];
    u_9 = u[IDG(3,1,ivg)];
    u10 = u[IDG(2,2,ivg)];
    u11 = u[IDG(3,2,ivg)];

    wt1r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt1i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
		     vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5);
    wt2r = MULT_GDXr(u_6, u_7, u_8, u_9, u10, u11,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);
    wt2i = MULT_GDXi(u_6, u_7, u_8, u_9, u10, u11,
		     vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5);

    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;
    
    ic = 2;
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

    v2L[  2*(ic+NCOL*0)] +=  wt1r;    v2L[1+2*(ic+NCOL*0)] +=  wt1i;
    v2L[  2*(ic+NCOL*1)] +=  wt2r;    v2L[1+2*(ic+NCOL*1)] +=  wt2i;

    // gm5_aypx and write back to global memory
    v2[IDX(0,2,ivx2)] = v1[IDX(0,0,ivx2)] - CKs * v2L[  2*(0+NCOL*0)];
    v2[IDX(1,2,ivx2)] = v1[IDX(1,0,ivx2)] - CKs * v2L[1+2*(0+NCOL*0)];
    v2[IDX(2,2,ivx2)] = v1[IDX(2,0,ivx2)] - CKs * v2L[  2*(1+NCOL*0)];
    v2[IDX(3,2,ivx2)] = v1[IDX(3,0,ivx2)] - CKs * v2L[1+2*(1+NCOL*0)];  
    v2[IDX(4,2,ivx2)] = v1[IDX(4,0,ivx2)] - CKs * v2L[  2*(2+NCOL*0)];
    v2[IDX(5,2,ivx2)] = v1[IDX(5,0,ivx2)] - CKs * v2L[1+2*(2+NCOL*0)];

    v2[IDX(0,3,ivx2)] = v1[IDX(0,1,ivx2)] - CKs * v2L[  2*(0+NCOL*1)];
    v2[IDX(1,3,ivx2)] = v1[IDX(1,1,ivx2)] - CKs * v2L[1+2*(0+NCOL*1)];
    v2[IDX(2,3,ivx2)] = v1[IDX(2,1,ivx2)] - CKs * v2L[  2*(1+NCOL*1)];
    v2[IDX(3,3,ivx2)] = v1[IDX(3,1,ivx2)] - CKs * v2L[1+2*(1+NCOL*1)];
    v2[IDX(4,3,ivx2)] = v1[IDX(4,1,ivx2)] - CKs * v2L[  2*(2+NCOL*1)];
    v2[IDX(5,3,ivx2)] = v1[IDX(5,1,ivx2)] - CKs * v2L[1+2*(2+NCOL*1)];
    
    v2[IDX(0,0,ivx2)] = v1[IDX(0,2,ivx2)] - CKs * v2L[  2*(0+NCOL*2)];
    v2[IDX(1,0,ivx2)] = v1[IDX(1,2,ivx2)] - CKs * v2L[1+2*(0+NCOL*2)];
    v2[IDX(2,0,ivx2)] = v1[IDX(2,2,ivx2)] - CKs * v2L[  2*(1+NCOL*2)];
    v2[IDX(3,0,ivx2)] = v1[IDX(3,2,ivx2)] - CKs * v2L[1+2*(1+NCOL*2)];
    v2[IDX(4,0,ivx2)] = v1[IDX(4,2,ivx2)] - CKs * v2L[  2*(2+NCOL*2)];
    v2[IDX(5,0,ivx2)] = v1[IDX(5,2,ivx2)] - CKs * v2L[1+2*(2+NCOL*2)];
    
    v2[IDX(0,1,ivx2)] = v1[IDX(0,3,ivx2)] - CKs * v2L[  2*(0+NCOL*3)];
    v2[IDX(1,1,ivx2)] = v1[IDX(1,3,ivx2)] - CKs * v2L[1+2*(0+NCOL*3)];
    v2[IDX(2,1,ivx2)] = v1[IDX(2,3,ivx2)] - CKs * v2L[  2*(1+NCOL*3)];
    v2[IDX(3,1,ivx2)] = v1[IDX(3,3,ivx2)] - CKs * v2L[1+2*(1+NCOL*3)];
    v2[IDX(4,1,ivx2)] = v1[IDX(4,3,ivx2)] - CKs * v2L[  2*(2+NCOL*3)];
    v2[IDX(5,1,ivx2)] = v1[IDX(5,3,ivx2)] - CKs * v2L[1+2*(2+NCOL*3)];
    
    ivx += blockDim.x * gridDim.x;
  }
}

void opr_DdagD_alt(real_t* __restrict__ v, real_t* __restrict__ u, 
		   real_t* __restrict__ w)
{
  static real_t *vt;
  static int STATIC_FLAG = 1;
  if(STATIC_FLAG) HANDLE_ERROR(cudaMalloc((void**)&vt, NVST2*sizeof(real_t)));

  MPI_Status st[10];
  MPI_Request req[10];

  int QCDSpinor_zyxvec = LZ * yx_Spinor;
  int QCDMatrix_zyxvec = LZ * yx_Matrix;
  int QCDSpinor_tyxvec = LT * yx_Spinor;
  int QCDMatrix_tyxvec = 4*LT*yx_Matrix;

  static real_t *tmp_QCDSpinor_s, *tmp_QCDSpinor_r;
  static real_t *tmp_QCDMatrix_s, *tmp_QCDMatrix_r;
  if(STATIC_FLAG) HANDLE_ERROR( cudaMalloc((void**)&tmp_QCDSpinor_s, 2*LT*yx_Spinor * sizeof(real_t)) );
  if(STATIC_FLAG) HANDLE_ERROR( cudaMalloc((void**)&tmp_QCDSpinor_r, 2*LT*yx_Spinor * sizeof(real_t)) );
  if(STATIC_FLAG) HANDLE_ERROR( cudaMalloc((void**)&tmp_QCDMatrix_s, 4*LT*yx_Matrix * sizeof(real_t)) );
  if(STATIC_FLAG) HANDLE_ERROR( cudaMalloc((void**)&tmp_QCDMatrix_r, 4*LT*yx_Matrix * sizeof(real_t)) );

#ifdef _PROF
  double tmp = dtime();
#endif
  pack_QCDMatrix <<< NUM_GANGS, VECTOR_LENGTH >>> (tmp_QCDMatrix_s, u);
  cudaDeviceSynchronize();
#ifdef _PROF
  prof_t[PACK] += dtime() - tmp;
  tmp = dtime();
#endif

  for(int i=0;i<4;i++){
    MPI_Irecv(u + (i*LT2*LZ2 + 1)*yx_Matrix,               QCDMatrix_zyxvec, MPI_DOUBLE, left,  i, comm_lr, req+0+i*2);
    MPI_Isend(u + (i*LT2*LZ2 + (LT2-2)*LZ2 + 1)*yx_Matrix, QCDMatrix_zyxvec, MPI_DOUBLE, right, i, comm_lr, req+1+i*2);
  }

  MPI_Irecv(tmp_QCDMatrix_r, QCDMatrix_tyxvec, MPI_DOUBLE, up,   5, comm_ud, req+8);
  MPI_Isend(tmp_QCDMatrix_s, QCDMatrix_tyxvec, MPI_DOUBLE, down, 5, comm_ud, req+9);

  MPI_Waitall(10, req, st);
#ifdef _PROF
  prof_t[COMM] += dtime() - tmp;
  tmp = dtime();
#endif
  unpack_QCDMatrix <<< NUM_GANGS, VECTOR_LENGTH >>> (u, tmp_QCDMatrix_r);
  pack_QCDSpinor <<< NUM_GANGS, VECTOR_LENGTH >>> (tmp_QCDSpinor_s, w, 0, 1);
  pack_QCDSpinor <<< NUM_GANGS, VECTOR_LENGTH >>> (tmp_QCDSpinor_s, w, 1, LZ2-2);
  cudaDeviceSynchronize();

#ifdef _PROF
  prof_t[PACK] += dtime() - tmp;
  tmp = dtime();
#endif
  MPI_Irecv(w + ((LT2-1)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, req+0);
  MPI_Irecv(w + yx_Spinor,                   QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, req+1);
  MPI_Isend(w + (LZ2 + 1)*yx_Spinor,         QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, req+2);
  MPI_Isend(w + ((LT2-2)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, req+3);
  
  MPI_Irecv(tmp_QCDSpinor_r + LT*yx_Spinor, QCDSpinor_tyxvec, MPI_DOUBLE, down, 12, comm_ud, req+4);
  MPI_Irecv(tmp_QCDSpinor_r,                QCDSpinor_tyxvec, MPI_DOUBLE, up,   13, comm_ud, req+5);
  MPI_Isend(tmp_QCDSpinor_s,                QCDSpinor_tyxvec, MPI_DOUBLE, up,   12, comm_ud, req+6);
  MPI_Isend(tmp_QCDSpinor_s + LT*yx_Spinor, QCDSpinor_tyxvec, MPI_DOUBLE, down, 13, comm_ud, req+7);
  
  MPI_Waitall(8, req, st);
#ifdef _PROF
  prof_t[COMM] += dtime() - tmp;
  tmp = dtime();
#endif
  unpack_QCDSpinor <<< NUM_GANGS, VECTOR_LENGTH >>> (w, tmp_QCDSpinor_r, 0, 0);
  unpack_QCDSpinor <<< NUM_GANGS, VECTOR_LENGTH >>> (w, tmp_QCDSpinor_r, LZ2-1, 1);
#ifdef _PROF
  cudaDeviceSynchronize();
  prof_t[PACK] += dtime() - tmp;
  tmp = dtime();
#endif
  opr_H_dev <<< NUM_GANGS, VECTOR_LENGTH >>> (vt, u, w);
#ifdef _PROF
  cudaDeviceSynchronize();
  prof_t[OPR] += dtime() - tmp;
  tmp = dtime();
#endif
  pack_QCDSpinor <<< NUM_GANGS, VECTOR_LENGTH >>> (tmp_QCDSpinor_s, vt, 0, 1);
  pack_QCDSpinor <<< NUM_GANGS, VECTOR_LENGTH >>> (tmp_QCDSpinor_s, vt, 1, LZ2-2);
  cudaDeviceSynchronize();

#ifdef _PROF
  prof_t[PACK] += dtime() - tmp;
  tmp = dtime();
#endif
  MPI_Irecv(vt + ((LT2-1)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 10, comm_lr, req+0);
  MPI_Irecv(vt + yx_Spinor,                   QCDSpinor_zyxvec, MPI_DOUBLE, left,  11, comm_lr, req+1);
  MPI_Isend(vt + (LZ2 + 1)*yx_Spinor,         QCDSpinor_zyxvec, MPI_DOUBLE, left,  10, comm_lr, req+2);
  MPI_Isend(vt + ((LT2-2)*LZ2 + 1)*yx_Spinor, QCDSpinor_zyxvec, MPI_DOUBLE, right, 11, comm_lr, req+3);
  
  MPI_Irecv(tmp_QCDSpinor_r + LT*yx_Spinor, QCDSpinor_tyxvec, MPI_DOUBLE, down, 12, comm_ud, req+4);
  MPI_Irecv(tmp_QCDSpinor_r,                QCDSpinor_tyxvec, MPI_DOUBLE, up,   13, comm_ud, req+5);
  MPI_Isend(tmp_QCDSpinor_s,                QCDSpinor_tyxvec, MPI_DOUBLE, up,   12, comm_ud, req+6);
  MPI_Isend(tmp_QCDSpinor_s + LT*yx_Spinor, QCDSpinor_tyxvec, MPI_DOUBLE, down, 13, comm_ud, req+7);
  
  MPI_Waitall(8, req, st);
#ifdef _PROF
  prof_t[COMM] += dtime() - tmp;
  tmp = dtime();
#endif
  unpack_QCDSpinor <<< NUM_GANGS, VECTOR_LENGTH >>> (vt, tmp_QCDSpinor_r, 0, 0);
  unpack_QCDSpinor <<< NUM_GANGS, VECTOR_LENGTH >>> (vt, tmp_QCDSpinor_r, LZ2-1, 1);
#ifdef _PROF
  cudaDeviceSynchronize();
  prof_t[PACK] += dtime() - tmp;
  tmp = dtime();
#endif
  opr_H_dev<<< NUM_GANGS, VECTOR_LENGTH >>>(v, u, vt);
  cudaDeviceSynchronize();
#ifdef _PROF
  prof_t[OPR] += dtime() - tmp;
#endif
  STATIC_FLAG = 0;
}
