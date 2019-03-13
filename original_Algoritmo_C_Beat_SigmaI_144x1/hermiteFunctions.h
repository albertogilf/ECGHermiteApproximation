#ifndef HERMITEFUNCTIONS_H_INCLUDED
#define HERMITEFUNCTIONS_H_INCLUDED
#define longitud(x)  (sizeof(x) / sizeof(x[0]))
#define PI 3.14159265358979323846
#define E 2.718281828459045235360
int factorial(int a);
double calc_phi(int index, float sigma, int orderPolynomial);
void calc_phis(float sigma, int currentPolynomialOrder, double *phis);
void calculatePhisTotal(float *phisTotal, float sigmaInit, float sigmaFinal);
float calculateQuadError(float* hermiteApproximation);
float calculateRelativeError(float *HermiteApproximation);
float calculateMaxError(float *hermiteApproximation);
float calculateVarError(float* hermiteApproximation);
float calculateError(float* hermiteApproximation);
float calculateDevError(float* hermiteApproximation);
void calculateHermiteApproximation(int sigma, float *hermiteApproximation, float *phisTotal, float *weightsHermitePolynomial);
void calculateWeightOfCoeficient(int sigmaIteration, int currentOrderPolynomial, float *weightsCurrentIteration, float *phisTotal);
float calculateWeightsOfPolynomial(float* a,float* results,float* phisTotal);
float hermiteApproximationOfTheHeartbeat(int* originalBeat, float frecuency, int orderPolynomial,float* results,float* weightsHermitePolynomial,float* phisTotal);
#endif // HERMITEFUNCTIONS_H_INCLUDED
