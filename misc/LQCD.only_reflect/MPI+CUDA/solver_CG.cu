/*
   CG Solver.
                             [2016.05.03 Hideo Matsufuru]
 */
#include "lattice.h"

static void solve_CG_init(real_t *__restrict__ rrp, real_t *__restrict__ rr, real_t *u, 
			  real_t *x, real_t *r, 
			  real_t *s, real_t *p)
{
#ifdef _PROF
  double tmp = dtime();
#endif
  copy(r, s);
  copy(x, s);
#ifdef _PROF
  cudaDeviceSynchronize();
  prof_t[COPY] += dtime() - tmp;
#endif
  opr_DdagD_alt(s, u, x);

#ifdef _PROF
  tmp = dtime();
#endif
  axpy(r, -1.0, s);
#ifdef _PROF
  cudaDeviceSynchronize();
  prof_t[AXPY] += dtime() - tmp;
  tmp = dtime();
#endif
  copy(p, r);
#ifdef _PROF
  cudaDeviceSynchronize();
  prof_t[COPY] += dtime() - tmp;
  tmp = dtime();
#endif
  *rrp = *rr = norm2(r);
#ifdef _PROF
  cudaDeviceSynchronize();
  prof_t[NORM] += dtime() - tmp;
#endif
}

static void solve_CG_step(real_t *__restrict__ rrp2, real_t *__restrict__ rr2, real_t *u, 
			  real_t *x, real_t *r, 
			  real_t *p, real_t *v)
{
  real_t rrp = *rrp2;

  opr_DdagD_alt(v, u, p);
#ifdef _PROF
  double tmp = dtime();
#endif
  real_t pap = dot(v, p);
#ifdef _PROF
  cudaDeviceSynchronize();
  prof_t[DOT] += dtime() - tmp;
#endif
  real_t cr = rrp/pap;
#ifdef _PROF
  tmp = dtime();
#endif
  axpy(x,  cr, p);
  axpy(r, -cr, v);
#ifdef _PROF
  cudaDeviceSynchronize();
  prof_t[AXPY] += dtime() - tmp;
  tmp = dtime();
#endif
  real_t rr = norm2(r);
#ifdef _PROF
  cudaDeviceSynchronize();
  prof_t[NORM] += dtime() - tmp;
#endif
  real_t bk = rr/rrp;
#ifdef _PROF
  tmp = dtime();
#endif
  scal(p, bk);
#ifdef _PROF
  cudaDeviceSynchronize();
  prof_t[SCAL] += dtime() - tmp;
  tmp = dtime();
#endif
  axpy(p, 1.0, r);
#ifdef _PROF
  cudaDeviceSynchronize();
  prof_t[AXPY] += dtime() - tmp;
#endif
  *rr2 = *rrp2 = rr;
}

void solve_CG(const real_t enorm, int *__restrict__ nconv, real_t *__restrict__ diff, real_t *xq, 
	      real_t *u, real_t *b)
{
  int niter = 1000;
  static real_t *x, *s, *r, *p;
  static int STATIC_FLAG = 1;
  if(STATIC_FLAG) HANDLE_ERROR( cudaMalloc((void**)&x, NVST2*sizeof(real_t)) );
  if(STATIC_FLAG) HANDLE_ERROR( cudaMalloc((void**)&s, NVST2*sizeof(real_t)) );
  if(STATIC_FLAG) HANDLE_ERROR( cudaMalloc((void**)&r, NVST2*sizeof(real_t)) );
  if(STATIC_FLAG) HANDLE_ERROR( cudaMalloc((void**)&p, NVST2*sizeof(real_t)) );
#ifdef _PROF
  double tmp = dtime();
#endif
  copy(s, b);
#ifdef _PROF
  cudaDeviceSynchronize();
  prof_t[COPY] += dtime() - tmp;
  tmp = dtime();
#endif
  real_t sr = norm2(s);
#ifdef _PROF
  cudaDeviceSynchronize();
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
  cudaDeviceSynchronize();
  prof_t[COPY] += dtime() - tmp;
#endif
  opr_DdagD_alt(r, u, x);
#ifdef _PROF
  tmp = dtime();
#endif
  axpy(r, -1.0, b);
#ifdef _PROF
  cudaDeviceSynchronize();
  prof_t[AXPY] += dtime() - tmp;
  tmp = dtime();
#endif
  *diff = norm2(r);
#ifdef _PROF
  cudaDeviceSynchronize();
  prof_t[NORM] += dtime() - tmp;
#endif
  STATIC_FLAG = 0;
}
