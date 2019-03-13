#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hermiteFunctions.h"
#include "hermite.h"

/* Author: Alberto Gil de la Fuente
   Purpose: ECG heartbeat approximation using Hermite Polynomials.
   See http://iwbbio.ugr.es/2014/papers/IWBBIO_2014_paper_60.pdf for more information
   Language: C
   mail: alberto.gilf@gmail.com
*/

//* Hermite order to compute
int maxOrderPolynomials = 6;
/**
     * Initial sigma value for 6th grade hermite polynomyals
     * 47ms / sample frecuency 
*/
float SIGMA0 = (float)47;
/**
     * sigma value for 6th grade hermite polynomyals
     */
float SIGMA = (float)1;
double error = 0;
double errorDev = 0;
double errorMax = 0;
float global_frequency;
// 72 samples per heartbeat in two channels = 144 samples
int Cnum_samples = 144;
int originalBeatInformation[144];
int numbeats = 1;

/* 
   Function: hermiteFunctions
   * Calculate the best sigma to approximate the heartbeat using the hermite polynomias 
   and fills the weights of the hermitePolynomial, the approximated beat and the phis
   * @param *originalBeat: contains the original samples of the heartbeat
   * @param frequency: the frequency used to take the input signal
   * @param orderPolynomial: grade of hermite polynomials used to calculate the approximated heartbeat
   * @param *approximatedBeat: array to write the approximated hearbeat using hermite polynomials
   * @param *weightsHermitePolynomial array to write the weights of each Hermite Polynomial
   * @param *phisTotal array to write the 
   * @returns: the value of the best sigma for the nth order Hermite Polynomial approximation of the heartbeat
*/
float hermiteApproximationOfTheHeartbeat(int* originalBeat, float frecuencia, int orderPolynomial, 
    float* approximatedBeat,float* weightsHermitePolynomial,float* phisTotal) 
{
    maxOrderPolynomials = orderPolynomial;
    float sigmaBest=0;
    switch (orderPolynomial) {
        case 6:
        SIGMA0 = 47;
        break;
        case 5:
        SIGMA0 = 51;
        break;
        case 4:
        SIGMA0 = 55;
        break;
        case 3:
        SIGMA0 = 62;
        break;
        default:
        maxOrderPolynomials = 6;
        SIGMA0 = 47;
    }

    global_frequency=frecuencia;
    int i=0;
    while(i<Cnum_samples)
    {
        originalBeatInformation[i]=originalBeat[i];
        i++;
    }
    sigmaBest=calculateWeightsOfPolynomial(weightsHermitePolynomial,approximatedBeat,phisTotal);
    return sigmaBest;
}

float calculateWeightsOfPolynomial(float* weightsHermitePoly,float* approximatedBeat,float* phisTotal) 
{
    float minError=100000;
    float currentError=0.0;
    float relativeError=0.0;
    float errorDev=0.0;
    float errorMax=0.0;
    float hermiteApproximation[numbeats*Cnum_samples];
    float weightsHermitePolynomialMinError[maxOrderPolynomials*numbeats];
    //double fiValueForHermitePolynomialsMinError[maxOrderPolynomials*Cnum_samples*numbeats];
    float sigmaFinal = SIGMA0 / (1000 / global_frequency);
    float sigmaInit = SIGMA / (1000 / global_frequency);
    float bestSigma;
    float currentSigma;
    int sigmaIteration=0;
    float weightsCurrentIteration[maxOrderPolynomials]; //Weights array
    int currentOrderPolynomials;
    // printf("\n calculating the weights of the hermitepolynomials for %f initial sigma and %f current sigma \n", sigmaFinal,currentSigma);
    //Find the best weights for the best possible values of Sigma
    for (currentSigma = sigmaInit; currentSigma <= sigmaFinal; currentSigma = currentSigma + sigmaInit)
    {
        // initialize at 0 the weights
        for (currentOrderPolynomials = 0; currentOrderPolynomials < maxOrderPolynomials; currentOrderPolynomials++)
        {
            weightsCurrentIteration[currentOrderPolynomials]=0.0;
        }
        for (currentOrderPolynomials = 0; currentOrderPolynomials < maxOrderPolynomials; currentOrderPolynomials++)
        {
            calculateWeightOfCoeficient(sigmaIteration,currentOrderPolynomials,weightsCurrentIteration,phisTotal);//Cálculo del peso para n
        }
        //  calculate error between the actual heartbeat and the hermite approximation
        calculateHermiteApproximation(sigmaIteration,hermiteApproximation,phisTotal, weightsCurrentIteration);
        currentError=calculateQuadError(hermiteApproximation);
        // Save the sigma, the current Hermite Approximation and the weights of the hermite polynomial 
        // if the error is lower than the previous ones (a better approximation of the heartbeat)
        if (currentSigma == sigmaInit || currentError < minError)
        {
            minError = currentError;
            for (currentOrderPolynomials = 0; currentOrderPolynomials < maxOrderPolynomials; currentOrderPolynomials++)
            {
                // Save the current weights to minimize the error
                weightsHermitePolynomialMinError[currentOrderPolynomials] = weightsCurrentIteration[currentOrderPolynomials];
            }
            // Save the best value for sigma
            bestSigma = currentSigma;
            // Save the hermiteAprroximation into the approximatedBeat 
            for(int i=0;i<Cnum_samples;i++)
            {
                approximatedBeat[i] = hermiteApproximation[i];
            }
        }
        sigmaIteration++;
    }
    relativeError=calculateRelativeError(approximatedBeat);
    errorDev=calculateDevError(approximatedBeat);
    errorMax=calculateMaxError(approximatedBeat);
    for (currentOrderPolynomials = 0; currentOrderPolynomials < maxOrderPolynomials; currentOrderPolynomials++)
    {
        weightsHermitePoly[currentOrderPolynomials]=weightsHermitePolynomialMinError[currentOrderPolynomials];
    }
    return bestSigma;
}

/* 
   Function: calculateWeightOfCoeficient
   * Fill the *weightsCurrentIteration depending on the corresponding phi. The phis are stored in *phisTotal. 
   The sigmaIteration and currentOrderPolynomial are used to accesses the proper index in the arrays.
   * @param sigmaIteration
   * @param currentOrderPolynomial
   * @param *weightsCurrentIteration
   * @param *phisTotal
*/
void calculateWeightOfCoeficient(int sigmaIteration, int currentOrderPolynomial, float *weightsCurrentIteration, float *phisTotal)
{
    int j = 0;
    int sigmaIndex = sigmaIteration * Cnum_samples * maxOrderPolynomials;
    int index = currentOrderPolynomial * Cnum_samples; // Current Polynomial Order (0-5) for number of samples (144)

    while (j < Cnum_samples)
    {
        weightsCurrentIteration[currentOrderPolynomial] += originalBeatInformation[j] * phisTotal[sigmaIndex + index + j];
        j++;
    }
}

/**
    Function: calculateHermiteApproximation
    * Calculate hermite Approximation of the heartbeat according to the sigma, the fis calculated based 
    on the sigma and the Weights of the HermitePolynomials
    * @param sigma Sigma to calculate the hermite Approximation
    * @param hermiteApproximation Array to fill with the values for the hermiteApproximation
    * @param phisTotal TOTAL phis
    * @param weightsHermitePolynomial Weights for each Hermite Polynomial
*/
void calculateHermiteApproximation(int sigma, float *hermiteApproximation, float *phisTotal, float *weightsHermitePolynomial)
{
    int i = 0;
    float valueHermiteApproximation = 0;
    int j = 0;
    int orderPolynomialIndex = 0; // para acceder al valor de phi de que cada polinomio de hermite (0 a 5) por numsignales(144)
    int phisIndex = sigma * maxOrderPolynomials * Cnum_samples;
    for (i = 0; i < Cnum_samples; i++)
    {
        valueHermiteApproximation = 0;
        for (j = 0; j < maxOrderPolynomials; j++)
        {
            orderPolynomialIndex = j * Cnum_samples;
            valueHermiteApproximation += weightsHermitePolynomial[j] * phisTotal[phisIndex + i + orderPolynomialIndex];
        }
        hermiteApproximation[i] = valueHermiteApproximation;
    }
}


/**
    Function: calculateQuadError
    * Calculate the quadratic error between the experimental value and its hermite approximation 
    * @param hermiteApproximation the values for the hermite approximation of the heartbeat
    * @returns the sum of the quadratic errors between the experimental heartbeat and the 
    hermite approximation
*/
float calculateQuadError(float *hermiteApproximation)
{
    float quadError = 0.0;
    int i = 0;
    for (i = 0; i < Cnum_samples; i++)
    {
        quadError += pow(originalBeatInformation[i] - hermiteApproximation[i], 2);
    }
    return quadError;
}

/**
    Function: calculateVarError
    * Calculate the variance error between the experimental value and its hermite approximation 
    * @param hermiteApproximation the values for the hermite approximation of the heartbeat
    * @returns the quadratic mean error between the experimental value and its hermite approximation 
    hermite approximation
*/
float calculateVarError(float* hermiteApproximation)
{
    float varError=0.0;
    float sumErrors=0.0;
    sumErrors=calculateError(hermiteApproximation);
    float avgError=sumErrors/Cnum_samples;
    float accumulatedError = 0;
    int i=0;
    for (i = 0; i < Cnum_samples; i++)
    {
        accumulatedError += pow(abs(originalBeatInformation[i] - hermiteApproximation[i]) - avgError, 2);
    }
    varError=accumulatedError / (Cnum_samples -1);
    return varError;
}

/**
    Function: calculateError
    * Calculate the sum of all the differences between 
    the experimental heartbeat and its hermite approximation 
    * @param hermiteApproximation the values for the hermite approximation of the heartbeat
    * @returns the sum of all errors between the experimental heartbeat and its hermite approximation 
    hermite approximation
*/
float calculateError(float* hermiteApproximation)
{
    float error=0.0;
    int i=0;
    for (i = 0; i < Cnum_samples; i++)
    {
        error += abs(originalBeatInformation[i] - hermiteApproximation[i]);
    }
    return error;
}

/**
    Function: calculateDevError
    * Counts the number of signals with a higher error than the double of the sqrt of 
    the variance
    * @param hermiteApproximation the values for the hermite approximation of the heartbeat
    * @returns the occurrences of the number of signals in the heartbeat with a higher error 
    than the double of the sqrt of the variance
*/
float calculateDevError(float* hermiteApproximation)
{
    float errorDev=0.0;
    float varError=0.0;
    varError=calculateVarError(hermiteApproximation);
    int i=0;
    for (i = 0; i < Cnum_samples; i++)
    {
        if (abs(originalBeatInformation[i] - hermiteApproximation[i]) > (sqrt(varError) * 2))
        {
            errorDev++;
        }
    }
    return errorDev;
}


/**
    Function: calculateMaxError
    * Calculates the maximum error between the original beat information and the hermite approximation
    * @param hermiteApproximation hermite approximation of the beat
    * @return the maximum Error
*/
float calculateMaxError(float *hermiteApproximation)
{
    float maxError = 0.0;
    int i = 0;
    float currentError = 0.0;
    for (i = 0; i < Cnum_samples; i++)
    {
        currentError = abs(originalBeatInformation[i] - hermiteApproximation[i]);
        if (currentError > maxError)
        {
            maxError = currentError;
        }
    }
    return maxError;
}


/**
    Function: calculateRelativeError
    * Calculates the relative error between the experimental values and the hermite approximation
    * @param HermiteApproximation hermite approximation of the beat
    * @return the relative Error
*/
float calculateRelativeError(float *HermiteApproximation)
{
    float relativeError = 0.0;
    float error = 0.0;
    error = calculateQuadError(HermiteApproximation);
    float totalSum = 0;
    int i = 0;
    for (i = 0; i < Cnum_samples; i++)
    {
        totalSum += (float)pow(originalBeatInformation[i], 2);
    }
    if (error == 0 || totalSum == 0)
    {
        relativeError = 0;
    }
    else
    {
        relativeError = error / totalSum;
    }
    return relativeError;
}

/* 
   Function: calc_phi
   * declarative implementation to calculate the phi of an index, a sigma an an order polynomial
   * @param index
   * @param sigma
   * @param orderPolynomial
   * @returns: the phi for the specificied sigma, orderPolynomial and index
*/

double calc_phi(int index, float sigma, int orderPolynomial)
{
    double denominator = sqrt(sigma * pow(2, orderPolynomial) * factorial(orderPolynomial) * sqrt(PI));
    double result = pow(E, -(pow(index, 2) / (2 * pow(sigma, 2)))) * HR(orderPolynomial, (index / sigma));
    result = result / denominator;
    return result;
}

/* 
   Function: calc_phis
   * Fill the array *phis with the phis for sigma sigma and the polynomial order polynomialOrder
   * @param sigma
   * @param polynomialOrder
   * @param *phis
*/

void calc_phis(float sigma, int currentPolynomialOrder, double *phis)
{
    int j = -Cnum_samples / 2;
    int h = currentPolynomialOrder * Cnum_samples; // polynomialOrder (0-5 for 6th order) for num of samples (72*2272)
    int index = 0; // index to save the info in the array *phis
    double phi = 0.0;
    int final = Cnum_samples / 2;
    while (j < final)
    {
        phi = calc_phi(j, sigma, currentPolynomialOrder);
        phis[h + index] = phi;
        j++;
        index++;
    }
}

/**
     Function: calculatePhisTotal
     * Calculate all the approx values for all Phis of the heartbeat
     * @param phisTotal Final values for Phis
     * @param sigmaInit initial sigma
     * @param sigmaFinal final sigma
     * @return
     */

void calculatePhisTotal(float *phisTotal, float sigmaInit, float sigmaFinal)
{
    int sigmaIndex = 0;
    int polynomialIndex = 0;
    int sampleIndex = 0;
    int sigmaIteration = 0;
    int sample = -Cnum_samples / 2;
    int final = Cnum_samples / 2;
    float currentSigma = 0.0;
    int currentOrderPolynomials = 0;
    for (currentSigma = sigmaInit; currentSigma <= sigmaFinal; currentSigma = currentSigma + (float)sigmaInit)
    {
        sigmaIndex = sigmaIteration * Cnum_samples * maxOrderPolynomials;
        for (currentOrderPolynomials = 0; currentOrderPolynomials < maxOrderPolynomials; currentOrderPolynomials++)
        {
            polynomialIndex = currentOrderPolynomials * Cnum_samples;
            float phi = 0.0;
            sample = -Cnum_samples / 2;
            sampleIndex = 0;
            while (sample < final)
            {
                phi = calc_phi(sample, currentSigma, currentOrderPolynomials);
                phisTotal[sigmaIndex + polynomialIndex + sampleIndex] = phi;
                sample++;
                sampleIndex++;
            }
        } // End polynomials loop
        sigmaIteration++;
    } // End sigmas loop
}

/**
    Function: factorial
    * Calculates the factorial of the number a
    * @param a the number whose factorial will be returned
    * @returns the factorial of the param a
*/
int factorial(int a)
{
    int resultado=1;
    while (a > 0)
    {
        resultado = resultado * a;
        a--;
    }
    return resultado;
    printf("\n \n resultado %d \n",resultado);
}