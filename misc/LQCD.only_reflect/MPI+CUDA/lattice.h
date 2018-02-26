#ifndef LATTICE_INCLUDED
#define LATTICE_INCLUDED
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

# define PT 2
# define PZ 8

# define NX      16
# define NY      NX
# define NZ      NX
# define NT      NX
# define NST     (NX*NY*NZ*NT)
# define NCOL    3
# define NVC     (NCOL*2)
# define NDF     (NCOL*NCOL*2)
# define ND      4
# define NVST    (NVC*ND*NST)
# define CKs     0.15

# define LT      (NT/PT)
# define LZ      (NZ/PZ)
# define LT2     ((LT)+2)
# define LZ2     ((LZ)+2)
# define NST2    (NX*NY*LZ2*LT2)
# define NVST2   (NVC*ND*NST2)
# define yx_Spinor (NY*NX*ND*NCOL*2)
# define yx_Matrix (NY*NX*NCOL*NCOL*2)
//# define NGPUS   4
typedef double real_t;

// solve.cu
void solve_CG(const real_t enorm, int *__restrict__ nconv, real_t *__restrict__ diff, real_t*  xq, real_t* u, real_t* b);

// field.cu
real_t dot(const real_t* __restrict__ v1, const real_t* __restrict__ v2);
void norm2_t(real_t* __restrict__ corr, const real_t* __restrict__ v1);
real_t norm2(const real_t* __restrict__ v);
void axpy(real_t* __restrict__ v, const real_t a, const real_t* __restrict__ w);
void scal(real_t* __restrict__ v, const real_t a);
void copy(real_t* __restrict__ v, const real_t* __restrict__ w);

// fopr_wilson_alt.cu
void create_cart(const int pt, const int zt, const int me);
void opr_DdagD_alt(double* __restrict__ v, double* __restrict__ u, double* __restrict__ w);

// MACROs
#define IDXV(x, y, ld)   ((x) + (y) * (ld))
#define IDX(ivc, id, iv) ((ivc) + (NVC)*(id + iv*ND))
#define IDG(ivc, ic, iv) ((ivc) + (NVC)*(ic + iv*NCOL))
#define MULT_GXr(u0,u1,u2,u3,u4,u5,v0,v1,v2,v3,v4,v5)   (u0*v0-u1*v1 + u2*v2-u3*v3 + u4*v4-u5*v5)
#define MULT_GXi(u0,u1,u2,u3,u4,u5,v0,v1,v2,v3,v4,v5)   (u0*v1+u1*v0 + u2*v3+u3*v2 + u4*v5+u5*v4)
#define MULT_GDXr(u0,u1,u2,u3,u4,u5,v0,v1,v2,v3,v4,v5)  (u0*v0+u1*v1 + u2*v2+u3*v3 + u4*v4+u5*v5)
#define MULT_GDXi(u0,u1,u2,u3,u4,u5,v0,v1,v2,v3,v4,v5)  (u0*v1-u1*v0 + u2*v3-u3*v2 + u4*v5-u5*v4)
#define EXT_IMG_R(v1r,v1i,v2r,v2i,w1r,w1i,w2r,w2i)      (v1r*w2r - v1i*w2i - v2r*w1r + v2i*w1i)
#define EXT_IMG_I(v1r,v1i,v2r,v2i,w1r,w1i,w2r,w2i)      (- v1r*w2i - v1i*w2r + v2r*w1i + v2i*w1r)

//#define NUM_GANGS 128
//#define VECTOR_LENGTH 256
//#define NUM_GANGS_OPR 64
//#define VECTOR_LENGTH_OPR 128
#define NUM_GANGS 56
#define VECTOR_LENGTH  128

static void HandleError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
    exit( EXIT_FAILURE );
  }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#ifdef _PROF
// For PROFILE
#define PROF_NUMS 8
#define PACK 0
#define COMM 1
#define OPR  2
#define COPY 3
#define AXPY 4
#define NORM 5
#define DOT  6
#define SCAL 7
extern double prof_t[PROF_NUMS];
double dtime();
#endif

#endif
