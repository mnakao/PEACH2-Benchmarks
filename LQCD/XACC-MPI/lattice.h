#ifndef LATTICE_INCLUDED
#define LATTICE_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <xmp.h>
#include <openacc.h>
# define PT 16
# define PZ 8

# define NX      32
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

# define NGPUS   4

typedef double real_t;

typedef struct QCDMatrix {
  real_t v[NCOL][NCOL][2];
} QCDMatrix_t;

typedef struct QCDSpinor {
  real_t v[ND][NCOL][2];
} QCDSpinor_t;

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

//#define NUM_GANGS 128
//#define VECTOR_LENGTH  256
#define NUM_GANGS 28
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
