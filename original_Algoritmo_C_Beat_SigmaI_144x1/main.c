
/* Author: Alberto Gil de la Fuente and Gabriel Caffarena (2019)
   Purpose: ECG heartbeat approximation using Hermite Polynomials.
   See http://iwbbio.ugr.es/2014/papers/IWBBIO_2014_paper_60.pdf for more information
   Language: C
   mail: alberto.gilf@gmail.com
*/

//includes, system

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

//includes, project
#include "hermiteFunctions.h"
#include "hermite.h"

//#define CHANNELS  2
#define NUMSIGMAS 47
#define NUMFILES 1 // it used to be 200
#define NUMSAMPLESHEARTBEAT 144

void loadsignals(int *channel1, int *channel2);
int loadHeartbeatcenters(int *heartbeatCenters, float *frequency, int *polynomials, int *channels, int *numSamples);
int calculateInitial(int heartbeatCenter, int *channel);
void loadHeartbeat(int initial, int *heartbeat, int *channel);
void loadAllHeartbeats(int *heartbeatCentersChannel1, int *heartbeatCentersChannel2,
   int *allHeartbeats, int *channel1, int *channel2, int numHeartbeats);
//void loadAllHeartbeatsNormalized(int *heartbeatCentersChannel1,int *heartbeatCentersChannel2,int *heartbeats,
//  int *channel1,int *channel2,int numHeartbeats);
void writeHermiteResults(float *hermiteApproximation, FILE *resultsFile, int numHeartbeats);
void writeWeightsResult(float *weightshermite, float *sigmaBest, FILE *resultsFile, int numHeartbeats, int polynomials);

/******** main *******
**********************/
int main()
{
    FILE *resultsFile = fopen(".//results//results.txt", "a+");
    FILE *errorsFile = fopen(".//results//errors.txt", "a+");

    int polynomials, channels, numSamples;
    float frequency;
    int heartbeatCenters[120000]; // GCF: oversized to fit 120,000 beats
    int heartbeatCenter = 0;
    int numHeartbeats;

    int *signalsChannel1; // signals measured in channel1
    int *signalsChannel2; // signal measured in channel 2

    float SIGMA0 = (float)1; // calculate initial sigma
    float sigma = (float)47; // calulate final sigma
    float sigmaInit;
    float sigmaFinal;

    float hermiteApproximation1[NUMSAMPLESHEARTBEAT];
    float hermiteApproximation2[NUMSAMPLESHEARTBEAT];
    float *sigmaBest1;
    float *sigmaBest2;
    float *totalHermiteApproximation1;
    float *totalHermiteApproximation2;
    float *phisTotal;
    float *weightsHermitePolynomial1;
    float *weightsHermitePolynomial2;
    float *totalWeightsPolynomial1;
    float *totalWeightsPolynomial2;

    int initial = 0; // initial position of the heartbeat
    int initial2 = 0;
    int weightsIndex = 0;
    int hermiteIndex = 0;
    int n = 0;
    int i = 0;

    int heartbeatChannel1[NUMSAMPLESHEARTBEAT]; // array containing all the relevant signals from channel1
    int heartbeatChannel2[NUMSAMPLESHEARTBEAT]; // array containing all the relevant signals from channel2

    resultsFile = fopen(".//results//results.txt", "a+");
    errorsFile = fopen(".//results//errors.txt", "a+");
    if (resultsFile == NULL || errorsFile == NULL)
    {
        printf("Error opening the results file of CPU");
        return -1;
    }

    numHeartbeats = loadHeartbeatcenters(heartbeatCenters, &frequency, &polynomials, &channels, &numSamples);
    printf("%d NUMSAMPLES %d CHANNELS %d POLYNOMIALS, %d HEARTBEATS, %d FILES, %f FREQ \n", numSamples, channels, polynomials, numHeartbeats, NUMFILES, frequency);

    // Set the initials sigmas to calculate the phi functions (freq is the variable)

    sigmaInit = SIGMA0 / (1000. / frequency);
    sigmaFinal = sigma / (1000. / frequency);

    /*
    Calculate the centers, adjust them, and save the signals I am interested in. 
    */

    signalsChannel1 = (int *)malloc(numSamples * sizeof(int));
    signalsChannel2 = (int *)malloc(numSamples * sizeof(int));
    loadsignals(signalsChannel1, signalsChannel2);

    sigmaBest1 = malloc(numHeartbeats * sizeof(float));
    sigmaBest2 = malloc(numHeartbeats * sizeof(float));
    totalHermiteApproximation1 = malloc(NUMSAMPLESHEARTBEAT * numHeartbeats * sizeof(float));
    totalHermiteApproximation2 = malloc(NUMSAMPLESHEARTBEAT * numHeartbeats * sizeof(float));
    ;
    phisTotal = malloc(NUMSIGMAS * polynomials * NUMSAMPLESHEARTBEAT * sizeof(float));
    weightsHermitePolynomial1 = malloc(polynomials * sizeof(float));
    weightsHermitePolynomial2 = malloc(polynomials * sizeof(float));
    totalWeightsPolynomial1 = malloc(polynomials * numHeartbeats * sizeof(float));
    totalWeightsPolynomial2 = malloc(polynomials * numHeartbeats * sizeof(float));

    calculatePhisTotal(phisTotal, sigmaInit, sigmaFinal);

    n = 0;

    while (n < numHeartbeats)
    {
        weightsIndex = n * polynomials;
        hermiteIndex = n * NUMSAMPLESHEARTBEAT;
        heartbeatCenter = heartbeatCenters[n];
        initial = calculateInitial(heartbeatCenter, signalsChannel1);
        initial2 = calculateInitial(heartbeatCenter, signalsChannel2);
        // FILL THE MATRIX WITH THE NORMALIZED HEARTBEAT AND THE NOISE FILTERED
        loadHeartbeat(initial, heartbeatChannel1, signalsChannel1);
        loadHeartbeat(initial2, heartbeatChannel2, signalsChannel2);

        sigmaBest1[n] = hermiteApproximationOfTheHeartbeat(heartbeatChannel1, frequency, polynomials, hermiteApproximation1, weightsHermitePolynomial1, phisTotal);
        sigmaBest2[n] = hermiteApproximationOfTheHeartbeat(heartbeatChannel2, frequency, polynomials, hermiteApproximation2, weightsHermitePolynomial2, phisTotal);

        i = 0;
        while (i < polynomials)
        {

            totalWeightsPolynomial1[weightsIndex + i] = weightsHermitePolynomial1[i];
            totalWeightsPolynomial2[weightsIndex + i] = weightsHermitePolynomial2[i];
            i++;
        }
        i = 0;
        while (i < NUMSAMPLESHEARTBEAT)
        {
            totalHermiteApproximation1[hermiteIndex + i] = hermiteApproximation1[i];
            totalHermiteApproximation2[hermiteIndex + i] = hermiteApproximation2[i];
            i++;
        }
        n++;
    }

    // Write hermite polynomial weights
    fputs("\n ########## channel 1: #########\n", resultsFile);
    writeWeightsResult(totalWeightsPolynomial1, sigmaBest1, resultsFile, numHeartbeats, polynomials);

    fputs("\n ########## channel 2: #########\n", resultsFile);
    writeWeightsResult(totalWeightsPolynomial2, sigmaBest2, resultsFile, numHeartbeats, polynomials);

    // Release memory of CPU
    n = 0;
    free(totalWeightsPolynomial1);
    free(totalWeightsPolynomial2);
    free(totalHermiteApproximation1);
    free(totalHermiteApproximation2);
    free(sigmaBest1);
    free(sigmaBest2);
    free(weightsHermitePolynomial1);
    free(weightsHermitePolynomial2);
    free(signalsChannel1);
    free(signalsChannel2);
    free(phisTotal);

    fclose(resultsFile);
    fclose(errorsFile);

    return 0;
}

/* 
   Function: loadHeartbeatcenters
   Load the beats based on the file prueba_Entry, where it is specified the: 
   line 0: frequency: 360 Hz
   line 1: Number of Hermite Polynomials:6
   line 2: Number of channels: 2
   line 3: Number of samples: 650,000
   From line 4 til end: approximate center of the heartbeats
   @param *heartbeatCenters array to save the centers of the heartbeats
   @param *frequency: pointer to int to save the frequency from the file
   @param *polynomials: pointer to int to save the number of hermite polynomials
   @param *channels: pointer to int to save the number of channels
   @param *polynomials: pointer to int to save the number of total signals (samples)
   @return: the number of heartbeats read in the input file
*/

int loadHeartbeatcenters(int *heartbeatCenters, float *frequency, int *polynomials, int *channels, int *numSamples)
{
    FILE *heartbeatCentersFile = fopen("data/prueba_Entry", "r");

    if (heartbeatCentersFile == NULL)
    {
        printf("Error opening the file of centers");
        return -1;
    }
    fscanf(heartbeatCentersFile, "%f", frequency);

    fscanf(heartbeatCentersFile, "%i", polynomials);

    fscanf(heartbeatCentersFile, "%i", channels);

    fscanf(heartbeatCentersFile, "%i", numSamples);
    //printf("**+ loadHeartbeatcenters --> numSamples= %d\n", numSamples); //DBG

    int numHeartbeats = 0;
    int indexHeartbeats = 0; // Read the heartbeat index in the file
    int heartbeatType = 0;
    while (fscanf(heartbeatCentersFile, "%d,%d,%d\n", &indexHeartbeats, &heartbeatType, &heartbeatCenters[numHeartbeats]) == 3)
    {
        numHeartbeats++;
    }
    fclose(heartbeatCentersFile);
    //printf("**+ loadHeartbeatcenters --> numSamples= %d\n", numSamples); //DBG

    return numHeartbeats;
}

/* 
   Function: loadsignals
   Reads the file where the signals are stored and save all of them in the 
   arrays *signalsChannel1 and *signalsChannel2
   @param *signalsChannel1 array to save the measurements of the channel 1
   @param *signalsChannel2 array to save the measurements of the channel 2
*/

void loadsignals(int *signalsChannel1, int *signalsChannel2)
{
    FILE *signalsFile = fopen("data/prueba_Signal", "r");
    if (signalsFile == NULL)
    {
        printf("Error  opening Signal file");
        return;
    }
    int signalIndex;
    int i = 0;
    int signal1;
    int signal2;
    /*
    Read all the signals, relevant and not relevant
    */
    //  printf("**+ Loadsignals before WHILE\n"); //DBG

    while (fscanf(signalsFile, "%d,%d,%d\n", &signalIndex, &signal1, &signal2) == 3)
    {
        //  printf( "%i, %d,%d,%d\n", i,signalIndex,signal1,signal2); //DBG
        signalsChannel1[i] = signal1;
        signalsChannel2[i] = signal2;
        i++;
    }
    fclose(signalsFile);
}

/* 
   Function: writeHermiteResults
   Write the heartbeat approximation calculated by the HermitePolynomialsApproximation 
   in the file results
   @param *hermiteApproximation: array where the approximated heartbeat is in memory. 
   It will be written in the file results
   @param *resultsFile: pointer to the file where the results are saved
   @param numHeartbeats: num of heartbeats
*/

void writeHermiteResults(float *hermiteApproximation, FILE *resultsFile, int numHeartbeats)
{
    int i = 0;
    int heartbeat = 0;
    //    int numSamplesHeartbeat=144;
    int indexHeartbeat = 0;
    int c;
    char car;
    int bufferPos = 0;
    long parteEntera, parteDecimal;
    float mul10 = 1;
    int precision = 9;
    char pBuffer[300];
    char bufHeartbeat[6];

    while (heartbeat < numHeartbeats)
    {
        fputs("\n heartbeat number: ", resultsFile);
        sprintf(bufHeartbeat, "%d", heartbeat);
        fputs(bufHeartbeat, resultsFile);
        fputs("\n aproximacion:", resultsFile);
        indexHeartbeat = heartbeat * NUMSAMPLESHEARTBEAT;
        for (i = 0; i < NUMSAMPLESHEARTBEAT; i++)
        {
            bufferPos = 0;
            mul10 = 1;
            for (c = 0; c < precision; c++)
            {
                mul10 *= 10;
            }
            parteEntera = abs((long)hermiteApproximation[indexHeartbeat + i]);
            parteDecimal = abs((long)((hermiteApproximation[indexHeartbeat + i] - (long)hermiteApproximation[indexHeartbeat + i]) * mul10));
            do
            {
                pBuffer[bufferPos] = (char)(parteEntera % 10) + '0';
                bufferPos++;
                parteEntera /= 10;
            } while (parteEntera > 0);
            if (hermiteApproximation[indexHeartbeat + i] < 0)
            {
                pBuffer[bufferPos] = '-';
                bufferPos++;
            }
            for (c = 0; c < bufferPos / 2; c++)
            {
                car = pBuffer[c];
                pBuffer[c] = pBuffer[bufferPos - c - 1];
                pBuffer[bufferPos - c - 1] = car;
            }
            if (precision > 0)
            {
                pBuffer[bufferPos] = '.';
                bufferPos++;
                int parteDecimalPos = bufferPos;
                for (c = 0; c < precision; c++)
                {
                    pBuffer[bufferPos] = (char)(parteDecimal % 10) + '0';
                    bufferPos++;
                    parteDecimal /= 10;
                }
                for (c = 0; c < precision / 2; c++)
                {
                    car = pBuffer[c + parteDecimalPos];
                    pBuffer[c + parteDecimalPos] = pBuffer[bufferPos - c - 1];
                    pBuffer[bufferPos - c - 1] = car;
                }
            }
            pBuffer[bufferPos] = 0;
            fputs(pBuffer, resultsFile);
            fputs(",", resultsFile);
        } // END SAMPLE HEARTBEATS LOOP
        heartbeat++;
    } // END HEARTBEATS LOOP
}

/* 
   Function: writeWeightsResult
   Write the weights and the best sigma for the heartbeat approximation calculated by the HermitePolynomialsApproximation 
   in the file resultsFile
   @param *weightshermite: array that contains the hermite weights
   @param *sigmaBest: array that contains the best sigma for the corresponding hermite weight
   @param *resultsFile: pointer to the file where the results are saved
   @param numHeartbeats: num of heartbeats
   @param polynomials: num of polynomials
*/

void writeWeightsResult(float *weightshermite, float *sigmaBest, FILE *resultsFile, int numHeartbeats, int polynomials)
{
    int i = 0;
    int heartbeat = 0;
    int indexHeartbeat = 0;
    int c;
    char car;
    int bufferPos = 0;
    long parteEntera, parteDecimal;
    float mul10 = 1;
    int precision = 9;
    char pBuffer[300];
    char bufHeartbeat[6];
    char bufsigma[6];
    char bufpolynomial[2];
    while (heartbeat < numHeartbeats)
    {
        fprintf(resultsFile, "\n heartbeat number: %d", heartbeat);
        fprintf(resultsFile, " best sigma: %f", sigmaBest[heartbeat]);
        indexHeartbeat = heartbeat * polynomials;
        for (i = 0; i < polynomials; i++)
        {
            fprintf(resultsFile, "\n polynomial number: %d \t", i);
            bufferPos = 0;
            mul10 = 1;
            for (c = 0; c < precision; c++)
            {
                mul10 *= 10;
            }
            parteEntera = abs((long)weightshermite[indexHeartbeat + i]);
            parteDecimal = abs((long)((weightshermite[indexHeartbeat + i] - (long)weightshermite[indexHeartbeat + i]) * mul10));
            do
            {
                pBuffer[bufferPos] = (char)(parteEntera % 10) + '0';
                bufferPos++;
                parteEntera /= 10;
            } while (parteEntera > 0);
            if (weightshermite[indexHeartbeat + i] < 0)
            {
                pBuffer[bufferPos] = '-';
                bufferPos++;
            }
            for (c = 0; c < bufferPos / 2; c++)
            {
                car = pBuffer[c];
                pBuffer[c] = pBuffer[bufferPos - c - 1];
                pBuffer[bufferPos - c - 1] = car;
            }
            if (precision > 0)
            {
                pBuffer[bufferPos] = '.';
                bufferPos++;
                int parteDecimalPos = bufferPos;
                for (c = 0; c < precision; c++)
                {
                    pBuffer[bufferPos] = (char)(parteDecimal % 10) + '0';
                    bufferPos++;
                    parteDecimal /= 10;
                }
                for (c = 0; c < precision / 2; c++)
                {
                    car = pBuffer[c + parteDecimalPos];
                    pBuffer[c + parteDecimalPos] = pBuffer[bufferPos - c - 1];
                    pBuffer[bufferPos - c - 1] = car;
                }
            }
            pBuffer[bufferPos] = 0;
            fputs(pBuffer, resultsFile);
        } // Heart polynomyal weights loop end
        heartbeat++;
    } // Heartbeat loop end
}

/* 
   Function: calculateInitial
   Load the heartbeat based on the center. It adjusts the real center (highest intensity signal) 
   and calculate the initial position of the hearbeat (QRS signals).
   @param heartbeatCenter Approximated center of the heartbeat
   @param *signals array that contains all the signals from the channel of interest
   @return the initial position of the hearbeat (QRS signals).
*/

int calculateInitial(int heartbeatCenter, int *signalsArray)
{
    int heartbeatInitial = heartbeatCenter - 36;
    int sumav1 = 0;
    int i = 0;
    float average;
    int max = 0;
    int maxPosition = 0;
    while (i < 72)
    {
        sumav1 += signalsArray[heartbeatInitial + i];
        i++;
    }
    average = sumav1 / 72;

    i = 0;
    while (i < 72)
    {
        if ((abs(signalsArray[heartbeatInitial + i] - average)) > max)
        {
            max = abs(signalsArray[heartbeatInitial + i] - average);
            maxPosition = heartbeatInitial + i;
        }
        i++;
    }
    return maxPosition - 36;
}

/* 
   Function: calculateAllCenters
   Load the heartbeats based on the center. It adjusts the real center (highest intensity signal) 
   and calculate the initial position of the hearbeats (QRS signals).
   @param *signals array that contains all the signals from the channel of interest
   @param *heartbeatCenters approximated centers of the heartbeats
   @param *newHeartbeatCenters adjusted centers of the heartbeats.
   @param numHeartbeats num of total heartbeats
*/
/*
void calculateAllCenters(int *signals, int *heartbeatCenters, int *newHeartbeatCenters, int numHeartbeats)
{
    int centerHeartbeat = 0;
    while (centerHeartbeat < numHeartbeats)
    {
        int initialHeartbeat = heartbeatCenters[centerHeartbeat] - 36;
        int sumav1 = 0;
        int i = 0;
        float average1;
        int max = 0;
        while (i < 72)
        {
            sumav1 += signals[initialHeartbeat + i];
            i++;
        }
        average1 = sumav1 / 72;

        i = 0;
        while (i < 72)
        {
            if ((abs(signals[initialHeartbeat + i] - average1)) > max)
            {
                max = abs(signals[initialHeartbeat + i] - average1);
                newHeartbeatCenters[centerHeartbeat] = (initialHeartbeat + i);
            }
            i++;
        }
        centerHeartbeat++;
    }
}
*/

/* 
   Function: loadHeartbeat
   Load the heartbeat based on the center and save the relevant signals (QRS) in *heartbeat. 
   It normalize the base signal in 0, and filters the noise.
   @param initial 
   @param *heartbeat array to save the relevant signals
   @param *signals array that contains all the signals from the channel of interest
*/

void loadHeartbeat(int initial, int *heartbeat, int *signals)
{
    int i = 0;
    int valueInInitIndex = signals[initial];
    int indexHeartbeat = 0; // If we load different heartbeats at the same time, we need an index
    int j = 0;              // To take the relevant signals after the real center is adjusted
    int max = 650000 - 72;
    // First heartbeat may be not completed at the beginning
    if (initial < 0)
    {
        initial = 0;
    }
    // Last heartbeat may be not completed at the end
    if (initial > max)
    {
        initial = max;
    }

    while (i < 36)
    {
        heartbeat[i + indexHeartbeat] = 0;
        i++;
    }
    while (i < 108)
    {
        heartbeat[i + indexHeartbeat] = signals[initial + j] - valueInInitIndex;
        i++;
        j++;
    }
    while (i < 144)
    {
        heartbeat[i + indexHeartbeat] = 0;
        i++;
    }
}

/* 
   Function: loadallHeartBeats
   Load the heartbeats based on the center and save the relevant signals (QRS) in *heartbeat. 
   It normalize the base signal in 0, and filters the noise.
   @param *heartbeatCentersChannel1 centers of heartbeats in channel1
   @param *heartbeatCentersChannel2 centers of heartbeats in channel2
   @param *allHeartbeats array to save the relevant signals from both channels
   @param *channel1 signals of heartbeats in channel1
   @param *channel2 signals of heartbeats in channel2
   @param numHeartbeats num of heartbeats in each channel
*/

void loadAllHeartbeats(int *heartbeatCentersChannel1, int *heartbeatCentersChannel2,
   int *allHeartbeats, int *channel1, int *channel2, int numHeartbeats)
{
    int heartbeat = 0;
    int indexHeartbeat; // calculate the index for all heartbeats
    int i = 0;
    int initial = 0;
    int valueInInitIndex = 0;
    int j = 0; // para coger los valores del array adecuados

    while (heartbeat < numHeartbeats)
    {
        initial = heartbeatCentersChannel1[heartbeat] - 35;
        i = 0;
        j = 0;
        valueInInitIndex = channel1[initial];
        indexHeartbeat = heartbeat * 144;
        int max = 650000 - 72;
        // First heartbeat may be not completed at the beginning
        if (initial < 0)
        {
            initial = 0;
        }
        // Last heartbeat may be not completed at the end
        if (initial > max)
        {
            initial = max;
        }
        while (i < 36)
        {
            allHeartbeats[indexHeartbeat + i] = 0;
            i++;
        }
        while (i < 108)
        {
            allHeartbeats[indexHeartbeat + i] = channel1[initial + j] - valueInInitIndex;
            i++;
            j++;
        }
        while (i < 144)
        {
            allHeartbeats[indexHeartbeat + i] = 0;
            i++;
        }
        heartbeat++;
    } // END LOOP HEARTBEATS

    // START THE READING OF THE SECOND CHANNEL

    heartbeat = 0;
    while (heartbeat < numHeartbeats)
    {
        initial = heartbeatCentersChannel2[heartbeat] - 35;
        i = 0;
        j = 0;
        valueInInitIndex = channel2[initial];
        indexHeartbeat = (numHeartbeats * 144) + (heartbeat * 144);
        int max = 650000 - 72;
        // First heartbeat may be not completed at the beginning
        if (initial < 0)
        {
            initial = 0;
        }
        // Last heartbeat may be not completed at the end
        if (initial > max)
        {
            initial = max;
        }
        while (i < 36)
        {
            allHeartbeats[i + indexHeartbeat] = 0;
            i++;
        }
        while (i < 108)
        {
            allHeartbeats[i + indexHeartbeat] = channel2[initial + j] - valueInInitIndex;
            i++;
            j++;
        }
        while (i < 144)
        {
            allHeartbeats[i + indexHeartbeat] = 0;
            i++;
        }
        heartbeat++;
    }
}

/* 
   Function: calculateInitialNormalized
   Calculate the initial position of the heartbeat using the absolute values
   @param *
   @param *heartbeat array to save the relevant signals
   @param *signals array that contains all the signals from the channel of interest
*/
/*
int calculateInitialNormalized(int heartbeatCenter, int *heartbeatSignals)
{
    int heartbeatInitial=heartbeatCenter-36;

    int i=0;

    int max=0;
    int maxPosition=0;

    i=0;
    while(i<72)
    {
        if(abs(channel[heartbeatInitial+i])>max)
        {
            max=abs(channel[heartbeatInitial+i]);
            maxPosition=heartbeatInitial+i;
        }
        i++;
    }
    return maxPosition-35;
}
*/

/* 
   Function: calculateAllCentersNormalized
   Calculate the initial position of the heartbeats using the absolute values
   @param *signals: the array of all signals taken in the corresponding channel
   @param *hertbeatCenters: array of the approximated centers
   @param *newHeartbeatCenters array to save the adjusted centers
   @param numHeartbeats: number of heartbeats
*/
/*
void calculateAllCentersNormalized(int* signals, int* hertbeatCenters,int* newHeartbeatCenters, int numHeartbeats)
{
  int heartbeatCenter=0;
  int i=0;
  int max=0;
  int heartbeat;
  while(heartbeatCenter<numHeartbeats)
  {
    heartbeat=hertbeatCenters[heartbeatCenter]-36;
    max=0;
    i=0;
    while(i<72)
    {
        if(abs(channel[heartbeat+i])>max)
        {
            max=abs(channel[heartbeat+i]);
            newHeartbeatCenters[heartbeatCenter]=(heartbeat+i);
        }
        i++;
    }
    heartbeatCenter++;
  }
}
*/

/* 
   Function: loadHeartbeatloadHeartbeatNormalized
   Load the heartbeat based on the center and save the relevant signals (QRS) in *heartbeat. 
   It normalize the base signal in 0, and filters the noise.
   @param initial 
   @param *heartbeat array to save the relevant signals
   @param *signals array that contains all the signals from the channel of interest
*/
/*
void loadHeartbeatNormalized(int initial,int *heartbeat,int *signalsChannel)
{
    int i=0;
    int indice=0; 
    int j=0; // para coger los valores del array adecuados
    int max=650000-72;
    // First heartbeat may be not completed at the beginning
    if(initial<0)
    {
        initial=0;
    }
    // Last heartbeat may be not completed at the end
    if (initial>max)
    {
        initial=max;
    }
    while(i<36)
    {
        heartbeat[i+indice]=0;
        i++;
    }
    while(i<108)
    {
        heartbeat[i+indice]=signalsChannel[initial+j];
        i++;
        j++;
    }
    while(i<144)
    {
        heartbeat[i+indice]=0;
        i++;
    }
}
*/

/* 
   Function: loadAllHeartbeatsNormalized
   Load the heartbeats based on the center and save the relevant signals (QRS) in *heartbeat. 
   It normalize the base signal in 0, and filters the noise.
   @param *heartbeatCentersChannel1 centers of heartbeats in channel1
   @param *heartbeatCentersChannel2 centers of heartbeats in channel2
   @param *allHeartbeats array to save the relevant signals from both channels
   @param *channel1 signals of heartbeats in channel1
   @param *channel2 signals of heartbeats in channel2
   @param numHeartbeats num of heartbeats in each channel
*/
/*
void loadAllHeartbeatsNormalized(int *heartbeatCentersChannel1,int *heartbeatCentersChannel2,
    int *heartbeats,int *signalsChannel1,int *signalsChannel2,int numHeartbeats)
{
    int heartbeat=0;
    int indice=0; // para hallar el indice y copiar todos los heartbeats
    int i=0;
    int initial=0;
    int j=0; // para coger los valores del array adecuados
    while(heartbeat<numHeartbeats)
    {
        initial=heartbeatCentersChannel1[heartbeat] - 35;
        i=0;
        j=0;
        indice=heartbeat*144;
        int max=650000-72;
        // First heartbeat may be not completed at the beginning
        if(initial<0)
        {
            initial=0;
        }
        // Last heartbeat may be not completed at the end
        if (initial>max)
        {
            initial=max;
        }
        while(i<36)
        {
            heartbeats[i+indice]=0;
            i++;
        }
        while(i<108)
        {
            heartbeats[i+indice]=signalsChannel1[initial+j];
            i++;
            j++;
        }
        while(i<144)
        {
            heartbeats[i+indice]=0;
            i++;
        }
        heartbeat++;
    } // END LOOP HEARTBEATS

    // Start to read the channel 2
    heartbeat=0;
    while(heartbeat<numHeartbeats)
    {
        initial=heartbeatCentersChannel2[heartbeat] - 35;
        i=0;
        j=0;
        indice=(numHeartbeats*144) + (heartbeat*144);
        int max=650000-72;
        // First heartbeat may be not completed at the beginning
        if(initial<0)
        {
            initial=0;
        }
        // Last heartbeat may be not completed at the end
        if (initial>max)
        {
            initial=max;
        }
        while(i<36)
        {
            heartbeats[i+indice]=0;
            i++;
        }
        while(i<108)
        {
            heartbeats[i+indice]=signalsChannel2[initial+j];
            i++;
            j++;
        }
        while(i<144)
        {
            heartbeats[i+indice]=0;
            i++;
        }
        heartbeat++;
    }
}

*/
