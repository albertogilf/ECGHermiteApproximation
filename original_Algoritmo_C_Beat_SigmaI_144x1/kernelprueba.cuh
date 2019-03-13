#ifndef HERMITEFUNCTIONS_H_INCLUDED
#define HERMITEFUNCTIONS_H_INCLUDED
#define longitud(x)  (sizeof(x) / sizeof(x[0]))
#define PI 3.14159265358979323846
#define E 2.718281828459045235360
extern void init();
__global__ void hermiteKernel(int *originalBeatGPU,float *weightsBestGPU,float *phisGPU, float *sigmaBestGPU,float sigmaInit, int numsigmas);
__device__ int factorialGPU(int a);
__device__ float HRGPU(int n, float x);
__global__ void phiKernel(float *fisGPU,float sigmaInit);
#endif // HERMITEFUNCTIONS_H_INCLUDED
