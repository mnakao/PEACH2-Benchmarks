#include "lattice.h"
extern QCDSpinor_t p[LT2][LZ2][NY][NX], x[LT2][LZ2][NY][NX];
#pragma acc declare present(p, x)

static void solve_CG_init(real_t *restrict rrp, real_t *restrict rr, QCDMatrix_t u[4][LT2][LZ2][NY][NX], 
			  QCDSpinor_t x[LT2][LZ2][NY][NX], QCDSpinor_t r[LT2][LZ2][NY][NX], 
			  QCDSpinor_t s[LT2][LZ2][NY][NX], QCDSpinor_t p[LT2][LZ2][NY][NX])
{
#ifdef _PROF
  double tmp = dtime();
#endif
  copy(r, s);
  copy(x, s);
#ifdef _PROF
  prof_t[COPY] += dtime() - tmp;
#endif
  opr_DdagD_alt(s, u, x, 0);
#ifdef _PROF
  tmp = dtime();
#endif
  axpy(r, -1.0, s);
#ifdef _PROF
  prof_t[AXPY] += dtime() - tmp;
  tmp = dtime();
#endif
  copy(p, r);
#ifdef _PROF
  prof_t[COPY] += dtime() - tmp;
  tmp = dtime();
#endif
  *rrp = *rr = norm2(r);
#ifdef _PROF
  prof_t[NORM] += dtime() - tmp;
#endif
}

static void solve_CG_step(real_t *restrict rrp2, real_t *restrict rr2, QCDMatrix_t u[4][LT2][LZ2][NY][NX], 
			  QCDSpinor_t x[LT2][LZ2][NY][NX], QCDSpinor_t r[LT2][LZ2][NY][NX], 
			  QCDSpinor_t p[LT2][LZ2][NY][NX], QCDSpinor_t v[LT2][LZ2][NY][NX])
{
  real_t rrp = *rrp2;

  opr_DdagD_alt(v, u, p, 1);
#ifdef _PROF
  double tmp = dtime();
#endif
  real_t pap = dot(v, p);
#ifdef _PROF
  prof_t[DOT] += dtime() - tmp;
#endif
  real_t cr = rrp/pap;
#ifdef _PROF
  tmp = dtime();
#endif
  axpy(x,  cr, p);
  axpy(r, -cr, v);
#ifdef _PROF
  prof_t[AXPY] += dtime() - tmp;
  tmp = dtime();
#endif
  real_t rr = norm2(r);
#ifdef _PROF
  prof_t[NORM] += dtime() - tmp;
#endif
  real_t bk = rr/rrp;
#ifdef _PROF
  tmp = dtime();
#endif
  scal(p, bk);
#ifdef _PROF
  prof_t[SCAL] += dtime() - tmp;
  tmp = dtime();
#endif
  axpy(p, 1.0, r);
#ifdef _PROF
  prof_t[AXPY] += dtime() - tmp;
#endif
  *rr2 = *rrp2 = rr;
}

void solve_CG(const real_t enorm, int *restrict nconv, real_t *restrict diff, QCDSpinor_t xq[LT2][LZ2][NY][NX], 
	      QCDMatrix_t u[4][LT2][LZ2][NY][NX], QCDSpinor_t b[LT2][LZ2][NY][NX])
{
  int niter = 1000;
  //  static QCDSpinor_t x[LT2][LZ2][NY][NX], s[LT2][LZ2][NY][NX], r[LT2][LZ2][NY][NX], p[LT2][LZ2][NY][NX];
  //#pragma acc enter data pcreate(x, s, r, p)
  static QCDSpinor_t s[LT2][LZ2][NY][NX], r[LT2][LZ2][NY][NX];
#pragma acc enter data pcreate(s, r)
#ifdef _PROF
  double tmp = dtime();
#endif
  copy(s, b);
#ifdef _PROF
  prof_t[COPY] += dtime() - tmp;
  tmp = dtime();
#endif
  real_t sr = norm2(s);
#ifdef _PROF
  prof_t[NORM] += dtime() - tmp;
#endif
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
    MPI_Finalize();
    exit(1);
  }

#ifdef _PROF
  tmp = dtime();
#endif
  copy(xq, x);
#ifdef _PROF
  prof_t[COPY] += dtime() - tmp;
#endif
  opr_DdagD_alt(r, u, x, 0);
#ifdef _PROF
  tmp = dtime();
#endif
  axpy(r, -1.0, b);
#ifdef _PROF
  prof_t[AXPY] += dtime() - tmp;
  tmp = dtime();
#endif
  *diff = norm2(r);
#ifdef _PROF
  prof_t[NORM] += dtime() - tmp;
#endif
}
