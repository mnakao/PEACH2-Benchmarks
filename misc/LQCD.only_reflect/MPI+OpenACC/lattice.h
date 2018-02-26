#ifndef LATTICE_INCLUDED
#define LATTICE_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include <string.h>
#include <openacc.h>

# define PT 2
# define PZ 8

# define NX      16
# define NY      NX
# define NZ      (NX)
# define NT      (NX)
# define NST     (NX*NY*NZ*NT)
# define NCOL    3
# define NVC     (NCOL*2)
# define NDF     (NCOL*NCOL*2)
# define ND      4
# define NVST    (NVC*ND*NST)
# define CKs     0.15

# define LT      (NT/PT)
# define LZ      (NZ/PZ)
# define LT2     (LT+2)
# define LZ2     (LZ+2)
# define NGPUS   1

# define yx_Spinor (NY*NX*ND*NCOL*2)
# define yx_Matrix (NY*NX*NCOL*NCOL*2)

typedef double real_t;

typedef struct QCDMatrix {
  real_t v[NCOL][NCOL][2];
} QCDMatrix_t;

typedef struct QCDSpinor {
  real_t v[ND][NCOL][2];
} QCDSpinor_t;

// solve.c
void solve_CG(const real_t enorm, int *restrict nconv, real_t *restrict diff, QCDSpinor_t xq[LT2][LZ2][NY][NX], QCDMatrix_t u[4][LT2][LZ2][NY][NX], QCDSpinor_t b[LT2][LZ2][NY][NX]);

// field.c
real_t dot(QCDSpinor_t v1[LT2][LZ2][NY][NX], QCDSpinor_t v2[LT2][LZ2][NY][NX]);
void norm2_t(real_t *restrict corr, QCDSpinor_t v[LT2][LZ2][NY][NX]);
real_t norm2(QCDSpinor_t v[LT2][LZ2][NY][NX]);
void axpy(QCDSpinor_t v[LT2][LZ2][NY][NX], real_t a, QCDSpinor_t w[LT2][LZ2][NY][NX]);
void scal(QCDSpinor_t v[LT2][LZ2][NY][NX], real_t a);
void copy(QCDSpinor_t v[LT2][LZ2][NY][NX], QCDSpinor_t w[LT2][LZ2][NY][NX]);

// fopr_wilson_alt.c
void create_cart(const int pt, const int zt, const int me);
void opr_DdagD_alt(QCDSpinor_t v[LT2][LZ2][NY][NX], QCDMatrix_t u[4][LT2][LZ2][NY][NX], QCDSpinor_t w[LT2][LZ2][NY][NX]);

// MACROs
#define MULT_GXr(u0,u1,u2,u3,u4,u5,v0,v1,v2,v3,v4,v5)   (u0*v0-u1*v1 + u2*v2-u3*v3 + u4*v4-u5*v5)
#define MULT_GXi(u0,u1,u2,u3,u4,u5,v0,v1,v2,v3,v4,v5)   (u0*v1+u1*v0 + u2*v3+u3*v2 + u4*v5+u5*v4)
#define MULT_GDXr(u0,u1,u2,u3,u4,u5,v0,v1,v2,v3,v4,v5)  (u0*v0+u1*v1 + u2*v2+u3*v3 + u4*v4+u5*v5)
#define MULT_GDXi(u0,u1,u2,u3,u4,u5,v0,v1,v2,v3,v4,v5)  (u0*v1-u1*v0 + u2*v3-u3*v2 + u4*v5-u5*v4)
#define EXT_IMG_R(v1r,v1i,v2r,v2i,w1r,w1i,w2r,w2i)      (v1r*w2r - v1i*w2i - v2r*w1r + v2i*w1i)
#define EXT_IMG_I(v1r,v1i,v2r,v2i,w1r,w1i,w2r,w2i)      (- v1r*w2i - v1i*w2r + v2r*w1i + v2i*w1r)

#define NUM_GANGS 56
#define VECTOR_LENGTH  128

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
