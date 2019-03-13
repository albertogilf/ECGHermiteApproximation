/*
 * RotationKernel.cu
 *
 *  Created on: May 09, 2013
 *      Author: Alberto
 *
 *
 */


//includes, kernel
#include <Kernelprueba.cu>


//includes, define
//#include "hermite.h"
#include "KernelHermite.h"

// Declaracion de variables en la memoria compartida __shared__ tipo nombre[<si es un array>]
// ejemplo:   __shared__ int s_puntos[TAMM];

    //* Funciones de Hermite a realizar --> 6
    int maxOrderPolynomialsGPU = 6;
    /**
     * Valor de sigmaGPU inicial para 6 funciones de hermite
     * 47ms divido por la frecuencyGPU de muestreo
     */
    float SIGMAGPU0 = (float) 47;
    /**
     * Valor de sigmaGPU para 6 funciones de hermite
     */
    float SIGMAGPU = (float) 1;
    float sigmaGPU = (float) 47;
    double errorGPU = 0;
    double errorGPUDev = 0;
    double errorGPUMax = 0;
    int originalBeatInformationCU[144];
    float frecuencyGPU=360.0;
    int nummuestrasGPU = 144;
    int numbeatsGPU=1;

    void HermiteFuctionsGPU(int* originalBeat, float frecuencia, int orderPolynomial, float* aproximateBeat,float* weightsHermitePolynomial) {
        switch (orderPolynomial) {
            case 6:
                SIGMAGPU0 = 47;
                break;
            case 5:
                SIGMAGPU0 = 51;
                break;
            case 4:
                SIGMAGPU0 = 55;
                break;
            case 3:
                SIGMAGPU0 = 62;
                break;
            default:
                maxOrderPolynomialsGPU = 6;
                SIGMAGPU0 = 47;
        }
        frecuencyGPU=frecuencia;
        int i=0;
        while(i<nummuestrasGPU)
        {
            originalBeatInformationCU[i]=originalBeat[i];
            i++;
        }
        calculateWeightsOfPolynomialGPU(weightsHermitePolynomial,aproximateBeat);
    }

        void calculateWeightsOfPolynomialGPU(float* weightsHermitePoly,float* aproximateBeat) {
        float minError=100000;
        int inicializacion=0;
        float currentError=0.0;
        float relativeError=0.0;
        float errorGPUDev=0.0;
        float errorGPUMax=0.0;
        float hermiteAproximation[Cmuestras];
        float weightsHermitePolynomialMinError[maxOrderPolynomialsGPU*numbeatsGPU];
        double fiValueForHermitePolynomialsMinError[maxOrderPolynomialsGPU*Cmuestras*numbeatsGPU];
        //@ duda 1000/360 antes supongo que dara mas o menos igual
        float sigmaFinal = SIGMAGPU0 / (1000 / frecuencyGPU);
        float sigmaInit = SIGMAGPU / (1000 / frecuencyGPU);
        sigmaGPU = 0;
        float currentSigma = sigmaGPUInit;
        //Buscamos los mejores pesos para los posibles valores de SIGMAGPU

            float weightsCurrentIteration[maxOrderPolynomialsGPU];//Vector de pesos para la iteración actual
            double fiValueForHermitePolynomialsCurrentIteration[maxOrderPolynomialsGPU * Cmuestras];//Matriz para los valores de fi para los maxOrderPolynomialsGPU valores de fi
            int currentOrderPolynomials = 0;
            int i=0;
            inicializacion=maxOrderPolynomialsGPU;
            for (i=0;i<inicializacion;i++)
            {
                weightsCurrentIteration[i]=0.0;
            }
            i=0;
            inicializacion=maxOrderPolynomialsGPU*Cmuestras;
            for (i=0;i<inicializacion;i++)
            {
                fiValueForHermitePolynomialsCurrentIteration[i]=0.0;
            }




            //Calculamos los pesos para los polinomios de Hermite
	    int *originalBeatGPU;
	    cudaMalloc((void**) &originalBeatGPU,144*sizeof(int));
	    cudaMemcpy(originalBeatGPU, originalBeatInformation, 144*sizeof(int),cudaMemcpyHostToDevice);
	    float *sigmaGPUErrorGPU;
	    cudaMalloc((void**) &sigmaGPUErrorGPU,47*sizeof(float));
            // CUDA MEMCOPY HOST TO DEVICE DEL LATIDO
	    float *sigmaError;
	    sigmaError= (float*) malloc(47*sizeof(float));
	    dim3 grid(2272,1);
	    dim3 bloque(144,1);
            //Calcular el errorGPU
	    HermiteKernel<<<grid,bloque>>>(originalBeatGPU,sigmaError,sigmaInit,sigmaFinal);

// El kernel calcula los fis, los weights, la aproximacion de hermite y el errorGPU para el sigmaGPU dado para cada punto.
	    

	    cudaMemcpy(sigmaGPUError, sigmaGPUErrorGPU, 144*sizeof(int),cudaMemcpyDeviceToHost);

            //Seleccionar el errorGPU más pequeño
            //@duda i==SIGMAGPU ¿?=¿?
            i=0;
            if (currentError < minError || currentSigma == sigmaGPUInit)
            {
                int m=0;
                minError = currentError;
                for(m=0;m<maxOrderPolynomialsGPU;m++)
                {
                    weightsHermitePolynomialMinError[m] = weightsCurrentIteration[m];//Almacenar los pesos que hacen que el errorGPU sea mínimo
                }

                sigmaGPU = currentSigma;//Almacenar el mejor valor de sigmaGPU
                m=0;
                int iteracion=maxOrderPolynomialsGPU*Cmuestras;
                int auxiliar=iteracion*i; // numero de latido y fi minima
                for(m=0;m<iteracion;m++)
                {
                    fiValueForHermitePolynomialsMinError[m+auxiliar] = fiValueForHermitePolynomialsCurrentIteration[m];
                }
        }/*
        calculateHermiteAproximationMinError(hermiteAproximation,fiValueForHermitePolynomialsMinError, weightsHermitePolynomialMinError);
        relativeError=calculateRelativeError(hermiteAproximation,fiValueForHermitePolynomialsMinError, weightsHermitePolynomialMinError);
        errorGPUDev=calculateErrorDev(hermiteAproximation,fiValueForHermitePolynomialsMinError, weightsHermitePolynomialMinError);
        errorGPUMax=calculateErrorMax(hermiteAproximation,fiValueForHermitePolynomialsMinError, weightsHermitePolynomialMinError);
        calculateAproximateBeat(hermiteAproximation,aproximateBeat);*/
        int m=0;
        int iteracion=maxOrderPolynomialsGPU;
        while(m<iteracion)
        {
            weightsHermitePoly[m]=weightsHermitePolynomialMinError[m];
            m++;
        }
    }

/*
     void calculateHermiteAproximation(float* hermiteAproximation,double* fiValuesForHermitePolynomials,float* weightsHermitePolynomial)
     {
        int i=0;
        float valueHermiteAproximation = 0;
        int j=0;
        int indice3=0;// para acceder al valor de fi de que cada polinomio de hermite (0 a 5) por numsignales(144)
        for (i = 0; i < Cmuestras; i++) // 0 a 144
        {
            valueHermiteAproximation = 0;
            for (j = 0; j < maxOrderPolynomialsGPU ; j++)
            {
                indice3=j*Cmuestras;
                valueHermiteAproximation += weightsHermitePolynomial[j] * fiValuesForHermitePolynomials[i+indice3];
            }
            hermiteAproximation[i]=valueHermiteAproximation;
        }
    }

    void calculateHermiteAproximationMinError(float* hermiteAproximation,double* fiValuesForHermitePolynomials,float* weightsHermitePolynomial)
    {
        int i=0;
        float valueHermiteAproximation = 0;
        int j=0;
        int indice3=0;// para acceder al valor de fi de que cada polinomio de hermite (0 a 5) por numsignales(72 *2272)
        for (i = 0; i < Cmuestras; i++)
        {
            valueHermiteAproximation = 0;
            for (j = 0; j < maxOrderPolynomialsGPU ; j++)
            {
                indice3=j*Cmuestras;
                valueHermiteAproximation += weightsHermitePolynomial[j] * fiValuesForHermitePolynomials[i+indice3];
            }
            hermiteAproximation[i]=valueHermiteAproximation;
        }
    }

    /**
     * Calculo del errorGPU entre los valores reales y la aproximación de hermite
     *
     * @param x Valores reales
     * @param fiValueForHermitePolynomials Valores de fi para cada polinomio de Hermite
     * @param weightsHermitePolynomial Pesos para cada polinomio de Hermite
     * @return Error
     */
/*
    float calculateError(float* hermiteAproximation,double* fiValuesForHermitePolynomials,float* weightsHermitePolynomial)
    {
        float currentError=0.0;
        int i=0;
        float errorGPUAcumulate = 0;
        for (i = 0; i < Cmuestras; i++)
        {
            errorGPUAcumulate += pow(originalBeatInformationCU[i] - hermiteAproximation[i], 2);
        }
        currentError=errorGPUAcumulate;
        return currentError;
    }

    float calculateErrorVar(float* hermiteAproximation,double* fiValuesForHermitePolynomials,float* weightsHermitePolynomial)
    {
        float errorGPUVar=0.0;
        float errorGPUMean=0.0;
        errorGPUMean=calculateErrorMean(hermiteAproximation,fiValuesForHermitePolynomials, weightsHermitePolynomial);
        float errorGPUAcumulate = 0;
        int i=0;
        for (i = 0; i < Cmuestras; i++)
        {
            errorGPUAcumulate += pow(abs(originalBeatInformationCU[i] - hermiteAproximation[i]) - errorGPUMean, 2);
        }
        errorGPUVar=errorGPUAcumulate * 1 / (nummuestrasGPU - 1);
        return errorGPUVar;

    }

     float calculateErrorMean(float* hermiteAproximation,double* fiValuesForHermitePolynomials, float* weightsHermitePolynomial)
     {
        float errorGPUMean=0.0;
        int i=0;
        for (i = 0; i < Cmuestras; i++)
        {
            errorGPUMean += abs(originalBeatInformationCU[i] - hermiteAproximation[i]);
        }
        return errorGPUMean;
    }


    float calculateErrorDev(float* hermiteAproximation,double* fiValuesForHermitePolynomials, float* weightsHermitePolynomial)
    {
        float errorGPUDev=0.0;
        float errorGPUVar=0.0;
        errorGPUVar=calculateErrorVar(hermiteAproximation,fiValuesForHermitePolynomials, weightsHermitePolynomial);
        int i=0;
        for (i = 0; i < Cmuestras; i++)
        {
            if (abs(originalBeatInformationCU[i] - hermiteAproximation[i]) > sqrt(errorGPUVar) * 2)
            {
                errorGPUDev++;
            }
        }
        return errorGPUDev;
    }

    float calculateErrorMax(float* hermiteAproximation,double* fiValuesForHermitePolynomials,float* weightsHermitePolynomial)
    {
        float errorGPUMax=0.0;
        int i=0;
        float errorGPUNew=0;
        for ( i = 0; i < Cmuestras; i++)
        {
            errorGPUNew=abs(originalBeatInformationCU[i] - hermiteAproximation[i]);
            if(errorGPUNew>errorGPUMax)
            {
                errorGPUMax=errorGPUNew;
            }
        }
        return errorGPUMax;

    }

    /**
     * Calculo del errorGPU entre los valores reales y la aproximación de hermite
     *
     * @param originalBeatInformationCU Valores reales
     * @param fiValuesForHermitePolynomials Valores de fi para cada polinomio de Hermite
     * @param weightsHermitePolynomial Pesos para cada polinomio de Hermite
     * @return Error
     */
/*
    float calculateRelativeError(float* HermiteAproximation,double* fiValuesForHermitePolynomials, float* weightsHermitePolynomial)
    {
        float relativeError=0.0;
        float errorGPU=0.0;
        errorGPU=calculateError(HermiteAproximation,fiValuesForHermitePolynomials, weightsHermitePolynomial);
        float totalSum = 0;
        int i=0;
        for ( i = 0; i < nummuestrasGPU; i++)
        {
            totalSum += (float) pow(float(originalBeatInformationCU[i]), 2);
        }
        if (errorGPU == 0 || totalSum == 0)
        {
            relativeError=0;
        } else
        {
            relativeError= errorGPU / totalSum;
        }
        return relativeError;
    }

    /**
     * Calcula los valores aproximados para un latido dado.
     *
     * @param originalBeatInformationCU
     * @param weightsHermitePolynomial
     * @param bestSigma
     * @return
     */
/*
    void calculateAproximateBeat(float* hermiteAproximation,float* aproximateBeat)
    {
        int i=0;
        for ( i = 0; i < Cmuestras; i++)
        {
            aproximateBeat[i]=hermiteAproximation[i];
        }
    }
*/

    /**
     * Calcula los valores aproximados para un latido dado.
     *
     * @param originalBeatInformationCU
     * @param weightsHermitePolynomial
     * @param bestSigma
     * @return
     */











