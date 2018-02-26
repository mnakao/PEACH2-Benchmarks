#include "lattice_reflect_ha_mpi.h"

__global__ static void _pack_QCDMatrix(real_t* __restrict__ v, const real_t* __restrict__ w)
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

__global__ static void _unpack_QCDMatrix(real_t* __restrict__ v, const real_t* __restrict__ w)
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

__global__ static void _pack_QCDSpinor(real_t* __restrict__ v, const real_t* __restrict__ w, const int v1, const int v2)
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

__global__ static void _unpack_QCDSpinor(real_t* __restrict__ v, const real_t* __restrict__ w, const int v1, const int v2)
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

void pack_QCDMatrix(QCDMatrix_t tmp_QCDMatrix_s[4][LT2-2][NY][NX], const QCDMatrix_t u[4][LT2][LZ2][NY][NX])
{
  _pack_QCDMatrix <<< NUM_GANGS, VECTOR_LENGTH >>> ((real_t*)tmp_QCDMatrix_s, (real_t*)u);
  cudaDeviceSynchronize();
}

void unpack_QCDMatrix(QCDMatrix_t u[4][LT2][LZ2][NY][NX], const QCDMatrix_t tmp_QCDMatrix_r[4][LT2-2][NY][NX])
{
  _unpack_QCDMatrix <<< NUM_GANGS, VECTOR_LENGTH >>> ((real_t*)u, (real_t*)tmp_QCDMatrix_r);
  cudaDeviceSynchronize();
}

void pack_QCDSpinor(QCDSpinor_t tmp_QCDSpinor_s[2][LT2-2][NY][NX], const QCDSpinor_t w[LT2][LZ2][NY][NX], const int ii, const int jj)
{
  _pack_QCDSpinor <<< NUM_GANGS, VECTOR_LENGTH >>> ((real_t*)tmp_QCDSpinor_s, (real_t*)w, ii, jj);
  cudaDeviceSynchronize();
}

void unpack_QCDSpinor(QCDSpinor_t w[LT2][LZ2][NY][NX], const QCDSpinor_t tmp_QCDSpinor_r[2][LT2-2][NY][NX], const int ii, const int jj)
{
  _unpack_QCDSpinor <<< NUM_GANGS, VECTOR_LENGTH >>> ((real_t*)w, (real_t*)tmp_QCDSpinor_r, ii, jj);
  cudaDeviceSynchronize();
}

