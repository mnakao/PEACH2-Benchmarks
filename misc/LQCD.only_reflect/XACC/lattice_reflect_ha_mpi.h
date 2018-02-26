#ifndef LATTICE_INCLUDED
#define LATTICE_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h> // unneeded
#ifdef _XCALABLEMP
#include <xmp.h>
#include <openacc.h>
#endif
#include <string.h> // unneeded
# define PT 1
# define PZ 1

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

// LT2 and LZ2 are unneeded
# define LT2     ((NT/PT)+2)
# define LZ2     ((NZ/PZ)+2)
//# define NGPUS   4

# define LT      (NT/PT)
# define LZ      (NZ/PZ)
# define yx_Spinor (NY*NX*ND*NCOL*2)
# define yx_Matrix (NY*NX*NCOL*NCOL*2)
#define IDXV(x, y, ld)   ((x) + (y) * (ld))

typedef double real_t;

typedef struct QCDMatrix {
  real_t v[NCOL][NCOL][2];
} QCDMatrix_t;

typedef struct QCDSpinor {
  real_t v[ND][NCOL][2];
} QCDSpinor_t;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
extern void pack_QCDMatrix(QCDMatrix_t tmp_QCDMatrix_s[4][LT2-2][NY][NX], const QCDMatrix_t u[4][LT2][LZ2][NY][NX]);
extern void unpack_QCDMatrix(QCDMatrix_t u[4][LT2][LZ2][NY][NX], const QCDMatrix_t tmp_QCDMatrix_r[4][LT2-2][NY][NX]);
extern void pack_QCDSpinor(QCDSpinor_t tmp_QCDSpinor_s[2][LT2-2][NY][NX], const QCDSpinor_t w[LT2][LZ2][NY][NX], const int ii, const int jj);
extern void unpack_QCDSpinor(QCDSpinor_t w[LT2][LZ2][NY][NX], const QCDSpinor_t tmp_QCDSpinor_r[2][LT2-2][NY][NX], const int ii, const int jj);
#ifdef __cplusplus
}
#endif /* __cplusplus */

// XcalableMP
#pragma xmp template t(0:NZ-1, 0:NT-1)
#pragma xmp nodes p(PZ,PT)
#pragma xmp distribute t(block, block) onto p

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
double prof_t[PROF_NUMS];
#endif

#endif
