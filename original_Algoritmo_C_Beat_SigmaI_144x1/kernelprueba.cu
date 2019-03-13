/*
 * kernelprueba.cu
 *
 *  Created on: May 12, 2013
 *      Author: Alberto Gil
 *
 *
 */
//#ifndef _KERNEL_H_
//#define _KERNEL_H_

#define FREQUENCY 360
#define NUMSAMPLESHEARTBEAT 144
#define HERMITEPOLYNOMIALSORDER 6
#define NUMSIGMAS 47
//#include "hermite.h"
#include "kernelprueba.cuh"
#include <math.h>

////////////////////////////////////////////////////////////////////////////////		
//	blockDim.x3 grid(BLK_NMBRS, BLK_NMBRS,1);
//	blockDim.x3 threads_BLQ(BLK_SIZE,1,1);
//
//	HermiteKernel<<grid,threads_BLQ>>(weightsHermitePoly, aproximateBeat);
////////////////////////////////////////////////////////////////////////////////

 __shared__ int s_originalBeatInformationGPU[NUMSAMPLESHEARTBEAT];
 __shared__ float s_hermiteApproximation[NUMSAMPLESHEARTBEAT];
 __shared__ float s_currentError[NUMSAMPLESHEARTBEAT];
 __shared__ double s_weightsBest[HERMITEPOLYNOMIALSORDER];
 __shared__ float s_weights[HERMITEPOLYNOMIALSORDER*NUMSAMPLESHEARTBEAT];

 __device__ int factorialGPU(int a)
 {
  int result=1;
  while (a > 0)
  {
    result = result * a;
    a--;
  }
  return result;
}


__device__ float HRGPU(int n, float x) {
  float result=0;
  if (n == 0) {
    result=1.00;
  } else if (n == 1) {
    result=2 * x;
  } else if (n == 2) {
    result=4 * (float) pow(x, 2) - 2;
  } else if (n == 3) {
    result=8 * (float) pow(x, 3) - 12 * x;
  } else if (n == 4) {
    result=16 * (float) pow(x, 4) - 48 * (float) pow(x, 2) + 12;
  } else if (n == 5) {
    result=32 * (float) pow(x, 5) - 160 * (float) pow(x, 3) + 120 * x;
  } else if (n == 6) {
    result=64 * (float) pow(x, 6) - 480 * (float) pow(x, 4) + 720 * (float) pow(x, 2) - 120;
  } else if (n == 7) {
    result=128 * (float) pow(x, 7) - 1344 * (float) pow(x, 5) + 3360 * (float) pow(x, 3) - 1680 * x;
  } else if (n == 8) {
    result=256 * (float) pow(x, 8) - 3584 * (float) pow(x, 6) + 13440 * (float) pow(x, 4) - 13440 * (float) pow(x, 2) + 1680;
  } else if (n == 9) {
    result=512 * (float) pow(x, 9) - 9216 * (float) pow(x, 7) + 48384 * (float) pow(x, 5) - 80640 * (float) pow(x, 3) + 30240 * x;
  } else {
    result=0;
  }
  return result;
}

__global__ void hermiteKernel(int *originalBeatGPU,float *weightsBestGPU,float *phisGPU, float *sigmaBestGPU,float sigmaInit, int numSigmas)
{
  int currentSigma=0;
  int j=0;
  float minError=100000000000.0;
  float sigmaBest;
  int idx = gridDim.x * blockDim.x * blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x;
  s_originalBeatInformationGPU[threadIdx.x]=originalBeatGPU[idx];
  float valueHermiteAproximation=0;
  for (currentSigma = 0; currentSigma < numSigmas; currentSigma++)
  {
    for(j=0;j<HERMITEPOLYNOMIALSORDER;j++)
    {
      s_weights[threadIdx.x+(j*blockDim.x)]=s_originalBeatInformationGPU[threadIdx.x] * phisGPU[currentSigma*blockDim.x*HERMITEPOLYNOMIALSORDER+blockDim.x*j+threadIdx.x];
    }
    for(unsigned int stride=128;stride>0;stride>>=1)
    {
      __syncthreads();
      if((threadIdx.x<stride) && ((threadIdx.x+stride)<blockDim.x))
      {
        s_weights[threadIdx.x]+=s_weights[threadIdx.x+stride];
        s_weights[threadIdx.x+NUMSAMPLESHEARTBEAT]+=s_weights[threadIdx.x+stride+NUMSAMPLESHEARTBEAT];
        s_weights[threadIdx.x+2*NUMSAMPLESHEARTBEAT]+=s_weights[threadIdx.x+stride+2*NUMSAMPLESHEARTBEAT];
        s_weights[threadIdx.x+3*NUMSAMPLESHEARTBEAT]+=s_weights[threadIdx.x+stride+3*NUMSAMPLESHEARTBEAT];
        s_weights[threadIdx.x+4*NUMSAMPLESHEARTBEAT]+=s_weights[threadIdx.x+stride+4*NUMSAMPLESHEARTBEAT];
        s_weights[threadIdx.x+5*NUMSAMPLESHEARTBEAT]+=s_weights[threadIdx.x+stride+5*NUMSAMPLESHEARTBEAT];
      }
    }
    __syncthreads();
    valueHermiteAproximation=0;
    for (j = 0; j < HERMITEPOLYNOMIALSORDER ; j++)
    {
      valueHermiteAproximation += s_weights[j*NUMSAMPLESHEARTBEAT] * phisGPU[currentSigma*blockDim.x*HERMITEPOLYNOMIALSORDER+(j*blockDim.x)+threadIdx.x];
    }
    s_hermiteApproximation[threadIdx.x]=valueHermiteAproximation;
    s_currentError[threadIdx.x] = pow((s_originalBeatInformationGPU[threadIdx.x] - s_hermiteApproximation[threadIdx.x]), 2);
    __syncthreads();

    for(unsigned int stride=128;stride>0;stride/=2)
    {
      __syncthreads();
      if((threadIdx.x<stride) && ((threadIdx.x+stride)<blockDim.x))
      {
        s_currentError[threadIdx.x]+=s_currentError[threadIdx.x+stride];
      }
    }
    __syncthreads();
    if(threadIdx.x==0)
    {
      if(s_currentError[threadIdx.x]<minError)
      {
        sigmaBest=currentSigma;
        minError=s_currentError[threadIdx.x];
        s_weightsBest[0]=s_weights[0];
        s_weightsBest[1]=s_weights[NUMSAMPLESHEARTBEAT];
        s_weightsBest[2]=s_weights[2*NUMSAMPLESHEARTBEAT];
        s_weightsBest[3]=s_weights[3*NUMSAMPLESHEARTBEAT];
        s_weightsBest[4]=s_weights[4*NUMSAMPLESHEARTBEAT];
        s_weightsBest[5]=s_weights[5*NUMSAMPLESHEARTBEAT];
      }
    }
  } // END SIGMA LOOP
  if(threadIdx.x<6)
  {
    if(threadIdx.x==0)
    {
      sigmaBestGPU[gridDim.x * blockIdx.y + blockIdx.x]=sigmaBest*sigmaInit+sigmaInit;
    }
    weightsBestGPU[(gridDim.x * blockIdx.y)*HERMITEPOLYNOMIALSORDER + blockIdx.x*HERMITEPOLYNOMIALSORDER + threadIdx.x]=s_weightsBest[threadIdx.x];
  }
} 
// END KERNEL

__global__ void phiKernel(float *phisGPU,float initSigma)
{
  float currentSigma = initSigma*blockIdx.x+initSigma; // Current sigma per block
  int sigmaIndex = blockIdx.x*NUMSAMPLESHEARTBEAT*gridDim.y;  
  int j = threadIdx.x - NUMSAMPLESHEARTBEAT / 2;
  int h = blockIdx.y*NUMSAMPLESHEARTBEAT; // indice orden polinomial
  double result = (pow(float(E), -(pow(float(j), float(2)) / (2 * pow(currentSigma, float(2))))) * HRGPU(int(blockIdx.y), 
    (j / currentSigma)))/sqrt(currentSigma * pow(float(2), int(blockIdx.y)) * factorialGPU(int(blockIdx.y)) * sqrt(PI));
  phisGPU[sigmaIndex+h+threadIdx.x]=result;
} 
// END KERNEL PHIS
