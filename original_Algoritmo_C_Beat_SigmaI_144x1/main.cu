


//includes, system
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <helper_timer.h>
//#include <cutil_inline.h>

// includes, project
extern "C" {
#include "hermite.h"
#include "hermiteFunctions.h"
}

//includes, kernel
#include "kernelprueba.cu"


#define CHANNELS  2
#define NUMSIGMAS 47
#define NUMFILES 1 // it used to be 200
#define NUMSAMPLESHEARTBEAT 144
#define NUMHEARTBEATS 135000

void loadsignals(int* signalsChannel1,int* signalsChannel2);
int loadHeartbeatcenters(int *heartbeatCenters,float *frequency_hb,int *polynomials, int *channels, int *numSamples);
int calculateInitial(int heartbeatCenter, int* signalsArray);
int calculateInitialNormalized(int heartbeatCenter, int *heartbeatSignals);
void calculateAllCenters(int *heartbeatSignals, int *heartbeatCenters, int *newHeartbeatCenters, int numHeartbeats);
void calculateAllCentersNormalized(int* heartbeatSignals, int* hertbeatCenters,int* newHeartbeatCenters, int numHeartbeats);
void loadHeartbeat(int initial, int *heartbeat, int *heartbeatSignals);
void loadAllHeartbeats(int *heartbeatCentersChannel1,int *heartbeatCentersChannel2, 
    int *allHeartbeats,int *heartbeatSignalsChannel1,int *heartbeatSignalsChannel2, int numHeartbeats);
void loadHeartbeatNormalized(int initial,int *heartbeat,int *signalsChannel);
void loadAllHeartbeatsNormalized(int *heartbeatCentersChannel1,int *heartbeatCentersChannel2,
    int *heartbeats,int *signalsChannel1,int *signalsChannel2,int numHeartbeats);
float errorMax(float *weights1CPU,float *weights2CPU, float *weightsGPU, 
    float *sigma1CPU, float *sigma2CPU, float *sigmaGPU, int numHeartbeats, int polynomials, FILE *errorsFile);
void writeHermiteResults(float *hermiteApproximation,FILE *resultsFile,int numHeartbeats);
void writeWeightsResult(float* weightshermite,float* sigmaBest,FILE *resultsFile,int numHeartbeats,int polynomials);
void writeWeightsResultGPU(float* weightshermite,float* sigmaBest, FILE *resultsFile,int numHeartbeats,int polynomials);

/******** main *******
**********************/
int main()
{
// timers to view serial and parallel time processing
    StopWatchInterface *timer_mallocmemory_serial = NULL;
    StopWatchInterface *timer_mallocmemory_parallel = NULL;
    StopWatchInterface *timer_readfile = NULL;
    StopWatchInterface *timer_calculate_serial = NULL;
    StopWatchInterface *timer_transferhtod_parallel = NULL;
    StopWatchInterface *timer_transferdtoh_parallel = NULL;
    StopWatchInterface *timer_calculate_parallel = NULL;
    StopWatchInterface *timer_writefile_serial = NULL;
    StopWatchInterface *timer_writefile_parallel = NULL;
    StopWatchInterface *timer_speedup = NULL;
    StopWatchInterface *timer_phis = NULL;
    StopWatchInterface *timer_load_beat_gpu = NULL;

    float time_mallocmemory_serial=0.0;
    float time_mallocmemory_parallel=0.0;
    float time_readfile=0.0;
    float time_calculate_serial=0.0;
    float time_transferhtod_parallel=0.0;
    float time_transferdtoh_parallel=0.0;
    float time_calculate_parallel=0.0;
    float time_writefile_serial=0.0;
    float time_writefile_parallel=0.0;
    float time_speedup=0.0;
    float time_phis=0.0;
    float time_load_beat_gpu=0.0;

    sdkCreateTimer(&timer_mallocmemory_serial);
    sdkCreateTimer(&timer_mallocmemory_parallel);
    sdkCreateTimer(&timer_readfile);
    sdkCreateTimer(&timer_calculate_serial);
    sdkCreateTimer(&timer_transferhtod_parallel);
    sdkCreateTimer(&timer_transferdtoh_parallel);
    sdkCreateTimer(&timer_calculate_parallel);
    sdkCreateTimer(&timer_writefile_serial);
    sdkCreateTimer(&timer_writefile_parallel);
    sdkCreateTimer(&timer_speedup);
    sdkCreateTimer(&timer_phis);
    sdkCreateTimer(&timer_load_beat_gpu);

// Create pointers to file to write the results and the errors.
    sdkStartTimer(&timer_readfile);
    FILE *resultsFile = fopen("/home/user/CUDA/ECG_original/original_Algoritmo_C_Beat_SigmaI_144x1/results/results.txt","a+");
    FILE *resultsFileGPU = fopen("/home/user/CUDA/ECG_original/original_Algoritmo_C_Beat_SigmaI_144x1/results/resultsGPU.txt","a+");
    FILE *errorsFile = fopen("/home/user/CUDA/ECG_original/original_Algoritmo_C_Beat_SigmaI_144x1/results/errors.txt","a+");
    if (resultsFile==NULL or resultsFileGPU==NULL or errorsFile==NULL)
    {
        printf("Error opening the results file of CPU and/or GPU");
        return -1;
    }
    char fileBuffer[10];

// read the heartbeat signals conditions
    int polynomials;
    float frequency_hb;
    int channels;
    int numSamples;
    int heartbeatCenters[NUMHEARTBEATS];
    int heartbeatCenter=0;
    int numHeartbeats;
    int *signalsChannel1; // signals measured in channel1
    signalsChannel1=(int *)malloc( 11000000*sizeof(int) );
    int *signalsChannel2; // signal measured in channel 2
    signalsChannel2=(int *)malloc( 1100000*sizeof(int) );
    numHeartbeats=loadHeartbeatcenters(heartbeatCenters, &frequency_hb, &polynomials, &channels, &numSamples);
    printf("%d HEARTBEATS %d TIMES \n",numHeartbeats,NUMFILES);

    // Set the initials sigmas to calculate the phis. The frequency_hb is a variable for it
    float SIGMA0 = (float) 1; // calculate initial sigma
    float sigma = (float) 47; // calulate final sigma
    float sigmaInit = SIGMA0 / (1000 / frequency_hb);
    float sigmaFinal = sigma / (1000 / frequency_hb);

    /*
    Calculate the centers, adjust them, and save the signals I am interested in. 
    */
    loadsignals(signalsChannel1,signalsChannel2);

    sdkStopTimer(&timer_readfile);
    time_readfile = sdkGetTimerValue(&timer_readfile);
    printf("EXECUTION TIME FOR READING FILES %f \n",time_readfile);

    // ALLOCATING HOST MEMORY
    sdkStartTimer(&timer_mallocmemory_serial);
    int heartbeatChannel1[NUMSAMPLESHEARTBEAT]; // array containing all the relevant signals from channel1
    int heartbeatChannel2[NUMSAMPLESHEARTBEAT]; // array containing all the relevant signals from channel2

    float hermiteApproximation1[NUMSAMPLESHEARTBEAT];
    float hermiteApproximation2[NUMSAMPLESHEARTBEAT];
    float sigmaBest1[numHeartbeats];
    float sigmaBest2[numHeartbeats];
    float totalHermiteApproximation1[NUMSAMPLESHEARTBEAT*numHeartbeats];
    float totalHermiteApproximation2[NUMSAMPLESHEARTBEAT*numHeartbeats];
    float phisTotal[NUMSIGMAS*polynomials*NUMSAMPLESHEARTBEAT];
    float weightsHermitePolynomial1[polynomials];
    float weightsHermitePolynomial2[polynomials];
    float totalWeightsPolynomial1[polynomials*numHeartbeats];
    float totalWeightsPolynomial2[polynomials*numHeartbeats];

    int initial=0; // initial position of the heartbeat
    int initial2=0;
    int weightsIndex=0;
    int hermiteIndex=0;
    int n=0;
    int i=0;
    sdkStopTimer(&timer_mallocmemory_serial);
    time_mallocmemory_serial = sdkGetTimerValue(&timer_mallocmemory_serial);

// CPU EXECUTION
    sdkStartTimer(&timer_calculate_serial);
// Calculate the phis
    calculatePhisTotal(phisTotal, sigmaInit, sigmaFinal);
    sdkStopTimer(&timer_calculate_serial);
    time_calculate_serial = sdkGetTimerValue(&timer_calculate_serial);
    int fileIndex=0;

    while(fileIndex<NUMFILES)
    {
        n=0;
        sdkResetTimer(&timer_calculate_serial);
        sdkStartTimer(&timer_calculate_serial);
        while (n<numHeartbeats)
        {
            i=0;
            weightsIndex=n*polynomials;
            hermiteIndex=n*NUMSAMPLESHEARTBEAT;
            heartbeatCenter=heartbeatCenters[n];
            initial=calculateInitial(heartbeatCenter,signalsChannel1);
            initial2=calculateInitial(heartbeatCenter,signalsChannel2);

            // FILL THE MATRIX WITH THE NORMALIZED HEARTBEAT AND THE NOISE FILTERED
            loadHeartbeat(initial,heartbeatChannel1,signalsChannel1);
            loadHeartbeat(initial2,heartbeatChannel2,signalsChannel2);
            sigmaBest1[n]=hermiteApproximationOfTheHeartbeat(heartbeatChannel1,frequency_hb,polynomials,hermiteApproximation1,weightsHermitePolynomial1,phisTotal);
            sigmaBest2[n]=hermiteApproximationOfTheHeartbeat(heartbeatChannel2,frequency_hb,polynomials,hermiteApproximation2,weightsHermitePolynomial2,phisTotal);
            while(i<polynomials)
            {
                totalWeightsPolynomial1[weightsIndex+i]=weightsHermitePolynomial1[i];
                totalWeightsPolynomial2[weightsIndex+i]=weightsHermitePolynomial2[i];
                i++;
            }
            i=0;
            while(i<NUMSAMPLESHEARTBEAT)
            {
                totalHermiteApproximation1[hermiteIndex+i]=hermiteApproximation1[i];
                totalHermiteApproximation2[hermiteIndex+i]=hermiteApproximation2[i];
                i++;
            }
            n++;
        }
        sdkStopTimer(&timer_calculate_serial);
        time_calculate_serial += sdkGetTimerValue(&timer_calculate_serial);

// WRITE CPU RESULTS FILE
        sdkResetTimer(&timer_writefile_serial);
        sdkStartTimer(&timer_writefile_serial);
        fputs("\n###################################### FILE ",resultsFile);
        sprintf(fileBuffer,"%d",resultsFile);
        fputs(fileBuffer,resultsFile);
        fputs(" ##############################\n \n",resultsFile);

        // Write hermite polynomial weights
        fputs("\n ########## channel 1: #########\n",resultsFile);
        writeWeightsResult(totalWeightsPolynomial1,sigmaBest1,resultsFile,numHeartbeats,polynomials);
        fputs("\n ########## channel 2: #########\n",resultsFile);
        writeWeightsResult(totalWeightsPolynomial2,sigmaBest2,resultsFile,numHeartbeats,polynomials);
    /*
        // WE DONT NEED TO WRITE THE APPROXIMATED BEAT SINCE WE WILL ONLY USE THE 
        // WEIGHTS OF THE POLYNOMIALS FOR EACH BEAT
        // WRITING THE APPROXIMATED HEARTBEAT
        fputs("\n ########## channel 1: #########\n",resultsFile);
        writeHermiteResults(totalHermiteApproximation1,resultsFile,numHeartbeats);
        fputs("\n ########## channel 2: ######### \n",resultsFile);
        writeHermiteResults(totalHermiteApproximation2,resultsFile,numHeartbeats);
    */
        sdkStopTimer(&timer_writefile_serial);
        time_writefile_serial += sdkGetTimerValue(&timer_writefile_serial);
        fileIndex++;
    }
    printf("CPU MEMORY ALLOCATION TIME %f \n",time_mallocmemory_serial);
    printf("CPU EXECUTION TIME %f \n",time_calculate_serial);
    printf("TOTAL TIME CPU %f \n",time_calculate_serial+time_mallocmemory_serial);
    printf("FILE WRITING CPU TIME %f \n",time_writefile_serial);
    
    // Not performed because they are used to write results on GPU and check errors
/*
    free(weightsHermitePolynomial1);
    free(weightsHermitePolynomial2);
    free(totalHermiteApproximation1);
    free(totalHermiteApproximation2);
    free(hermiteApproximation1);
    free(hermiteApproximation2);
    free(heartbeatChannel1);
    free(heartbeatChannel2);
    free(totalWeightsPolynomial1);
    free(totalWeightsPolynomial2);
    free(sigmaBest1);
    free(sigmaBest2);
  */

    sdkStartTimer(&timer_speedup);
    sdkStartTimer(&timer_phis);
    // ALLOCATE MEMORY FOR PHIS IN GPU
    float *phisGPU;
    cudaMalloc((void**) &phisGPU,NUMSAMPLESHEARTBEAT*polynomials*NUMSIGMAS*sizeof(float));

    // float *phis;
    // phis= (float*) malloc(NUMSAMPLESHEARTBEAT*polynomials*NUMSIGMAS*sizeof(float));

    // PHIS KERNEL DIMENSION
    dim3 gridfi(NUMSIGMAS,polynomials);
    dim3 blockPhi(NUMSAMPLESHEARTBEAT,1);
    //Calculate the phis in GPU
    phiKernel<<<gridfi,blockPhi>>>(phisGPU,sigmaInit);
    cudaDeviceSynchronize();

// DONT TRANSFER DATA FROM GPU TO CPU TO USE IT IN THE NEXT KERNEL
// DONT RELEASE MEM IN GPU TO USE IT IN THE NEXT KERNEL
// cudaMemcpy(phis, phisGPU, NUMSAMPLESHEARTBEAT*polynomials*NUMSIGMAS*sizeof(float),cudaMemcpyDeviceToHost);
    sdkStopTimer(&timer_phis);
    time_phis = sdkGetTimerValue(&timer_phis);

// ALLOCATING MEMORY FOR GPU EXECUTION
    sdkStartTimer(&timer_mallocmemory_parallel);
    int heartbeatCentersChannel1[numHeartbeats];
    int heartbeatCentersChannel2[numHeartbeats];
    int allHeartbeats[NUMSAMPLESHEARTBEAT*numHeartbeats*CHANNELS]; // array for all the relevant signals

// ALLOCATING CPU MEMORY TO TRANSFER RESULT FROM CPU
    float *weightsBest;
    weightsBest= (float*) malloc(polynomials*numHeartbeats*CHANNELS*sizeof(float));
    float *sigmaBest;
    sigmaBest= (float*) malloc(numHeartbeats*CHANNELS*sizeof(float));

    // ALLOCATE MEMORY IN GPU FOR ORIGINAL BEATS
    int *originalBeatGPU;
    cudaMalloc((void**)&originalBeatGPU,NUMSAMPLESHEARTBEAT*numHeartbeats*CHANNELS*sizeof(int));
    int *originalBeatGPU2;
    cudaMalloc((void**)&originalBeatGPU2,NUMSAMPLESHEARTBEAT*numHeartbeats*CHANNELS*sizeof(int));
    float *weightsBestGPU;
    cudaMalloc((void**) &weightsBestGPU,polynomials*numHeartbeats*CHANNELS*sizeof(float));
    float *sigmaBestGPU;
    cudaMalloc((void**) &sigmaBestGPU,numHeartbeats*CHANNELS*sizeof(float));

    // DIMENSIONS OF GRID AND BLOCKS
    dim3 grid(numHeartbeats,CHANNELS);
    dim3 block(NUMSAMPLESHEARTBEAT,1);

    sdkStopTimer(&timer_mallocmemory_parallel);
    time_mallocmemory_parallel += sdkGetTimerValue(&timer_mallocmemory_parallel);

    // LOAD THE ARRAY OF THE CENTERS
    calculateAllCenters(signalsChannel1,heartbeatCenters,heartbeatCentersChannel1,numHeartbeats);
    calculateAllCenters(signalsChannel2,heartbeatCenters,heartbeatCentersChannel2,numHeartbeats);
    
    // LOAD ALL HEARTBEATS
    loadAllHeartbeats(heartbeatCentersChannel1,heartbeatCentersChannel2,allHeartbeats,signalsChannel1,signalsChannel2,numHeartbeats);
    sdkStopTimer(&timer_load_beat_gpu);
    time_load_beat_gpu = sdkGetTimerValue(&timer_load_beat_gpu);

    // SENT THE FIRST HEARTBEAT
    sdkStartTimer(&timer_transferhtod_parallel);
    cudaMemcpy(originalBeatGPU, allHeartbeats, NUMSAMPLESHEARTBEAT*numHeartbeats*CHANNELS*sizeof(int),cudaMemcpyHostToDevice);
    sdkStopTimer(&timer_transferhtod_parallel);
    time_transferhtod_parallel += sdkGetTimerValue(&timer_transferhtod_parallel);

    sdkStopTimer(&timer_speedup);
    time_speedup += sdkGetTimerValue(&timer_speedup);

    fileIndex=0;
    while(fileIndex<NUMFILES)
    {
        sdkResetTimer(&timer_speedup);
        sdkStartTimer(&timer_speedup);
// WE DO THE LOOP UNROLLING TO LOAD THE NEXT BEATS IN GPU WHILE THE FIRST ONES ARE BEING PROCESSED
        if(fileIndex%2==0)
        {
            sdkResetTimer(&timer_calculate_parallel);
            sdkStartTimer(&timer_calculate_parallel);

            hermiteKernel<<<grid,block>>>(originalBeatGPU,weightsBestGPU,phisGPU,sigmaBestGPU,sigmaInit,NUMSIGMAS);

            sdkResetTimer(&timer_load_beat_gpu);
            sdkStartTimer(&timer_load_beat_gpu);
            // LOAD THE ARRAY OF CENTERS
            calculateAllCenters(signalsChannel1,heartbeatCenters,heartbeatCentersChannel1,numHeartbeats);
            calculateAllCenters(signalsChannel2,heartbeatCenters,heartbeatCentersChannel2,numHeartbeats);
            // LOAD THE ARRAY OF SIGNALS

            loadAllHeartbeats(heartbeatCentersChannel1,heartbeatCentersChannel2,allHeartbeats,signalsChannel1,signalsChannel2,numHeartbeats);

            sdkStopTimer(&timer_load_beat_gpu);
            time_load_beat_gpu += sdkGetTimerValue(&timer_load_beat_gpu);

            sdkResetTimer(&timer_transferhtod_parallel);
            sdkStartTimer(&timer_transferhtod_parallel);
            cudaMemcpy(originalBeatGPU2, allHeartbeats, NUMSAMPLESHEARTBEAT*numHeartbeats*CHANNELS*sizeof(int),cudaMemcpyHostToDevice);
            sdkStopTimer(&timer_transferhtod_parallel);
            time_transferhtod_parallel += sdkGetTimerValue(&timer_transferhtod_parallel);
        }
        else
        {
            sdkResetTimer(&timer_calculate_parallel);
            sdkStartTimer(&timer_calculate_parallel);

            hermiteKernel<<<grid,block>>>(originalBeatGPU2,weightsBestGPU,phisGPU,sigmaBestGPU,sigmaInit,NUMSIGMAS);
            sdkResetTimer(&timer_load_beat_gpu);
            sdkStartTimer(&timer_load_beat_gpu);
            // LOAD THE ARRAY OF CENTERS
            calculateAllCenters(signalsChannel1,heartbeatCenters,heartbeatCentersChannel1,numHeartbeats);
            calculateAllCenters(signalsChannel2,heartbeatCenters,heartbeatCentersChannel2,numHeartbeats);
            
            // LOAD THE ARRAY OF SIGNALS
            loadAllHeartbeats(heartbeatCentersChannel1,heartbeatCentersChannel2,allHeartbeats,signalsChannel1,signalsChannel2,numHeartbeats);

            sdkStopTimer(&timer_load_beat_gpu);
            time_load_beat_gpu += sdkGetTimerValue(&timer_load_beat_gpu);

            sdkResetTimer(&timer_transferhtod_parallel);
            sdkStartTimer(&timer_transferhtod_parallel);
            cudaMemcpy(originalBeatGPU, allHeartbeats, NUMSAMPLESHEARTBEAT*numHeartbeats*CHANNELS*sizeof(int),cudaMemcpyHostToDevice);
            sdkStopTimer(&timer_transferhtod_parallel);
            time_transferhtod_parallel += sdkGetTimerValue(&timer_transferhtod_parallel);
        }

        cudaDeviceSynchronize();
        sdkStopTimer(&timer_calculate_parallel);
        time_calculate_parallel += sdkGetTimerValue(&timer_calculate_parallel);
// TRANFER DATA FROM GPU TO HOST
        sdkResetTimer(&timer_transferdtoh_parallel);
        sdkStartTimer(&timer_transferdtoh_parallel);

        cudaMemcpy(weightsBest, weightsBestGPU, polynomials*numHeartbeats*CHANNELS*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(sigmaBest, sigmaBestGPU, numHeartbeats*CHANNELS*sizeof(float),cudaMemcpyDeviceToHost);
        sdkStopTimer(&timer_transferdtoh_parallel);
        time_transferdtoh_parallel += sdkGetTimerValue(&timer_transferdtoh_parallel);
// FREE GPU MEMORY
        sdkStopTimer(&timer_speedup);
        time_speedup += sdkGetTimerValue(&timer_speedup);

// WRITE IN FILES THE RESULTS OF PARALLEL EXECUTION
        sdkResetTimer(&timer_writefile_parallel);
        sdkStartTimer(&timer_writefile_parallel);
        fputs("\n \n###################################### FILE ",resultsFileGPU);
        sprintf(fileBuffer,"%d",fileIndex);
        fputs(fileBuffer,resultsFileGPU);
        fputs(" ##############################\n \n",resultsFileGPU);
        writeWeightsResultGPU(weightsBest,sigmaBest,resultsFileGPU,numHeartbeats,polynomials);

        sdkStopTimer(&timer_writefile_parallel);
        time_writefile_parallel += sdkGetTimerValue(&timer_writefile_parallel);
        fileIndex++;
    }

    cudaFree(originalBeatGPU);
    cudaFree(weightsBestGPU);
    cudaFree(sigmaBestGPU);
    cudaFree(phisGPU);

    float maxError=0.0;
    maxError=errorMax(totalWeightsPolynomial1,totalWeightsPolynomial2, weightsBest, sigmaBest1, sigmaBest2, sigmaBest, numHeartbeats,polynomials,errorsFile);
// Release memory of CPU
    // NOT USED BECAUSE WE USE DEF
    //free(heartbeatCentersChannel1);
    //free(heartbeatCentersChannel2);
    //free(allHeartbeats);
    //free(weightsBest);
    //free(sigmaBest);

    printf("\n EXECUTION TIME OF PHIS CALCULATION IN GPU %f \n",time_phis);
    printf("GPU MEMORY ALLOCATION TIME %f \n",time_mallocmemory_parallel);
    //     printf("GPU LOAD HEARTBEATS TIME %f \n",time_load_beat_gpu);
    printf("DATA TRANSFER MEMORY CPU TO GPU TIME %f \n",time_transferhtod_parallel);
    printf("KERNEL EXECUTION TIME IN GPU %f \n",time_calculate_parallel);
    printf("DATA TRANSFER MEMORY GPU TO CPU TIME%f \n",time_transferdtoh_parallel);
    printf("TOTAL EXECUTION TIME GPU %f SPEEDPUP %f \n",time_speedup,time_calculate_serial,time_calculate_serial/time_speedup);
    printf("WRITING RESULT FILES OF GPU TIME %f \n",time_writefile_parallel);

// return 0 of no errors
    return 0;
}

/** 
   Function: loadHeartbeatcenters
   * Loads the beats based on the file prueba_Entry, where it is specified the: 
   line 0: frequency_hb: 360 Hz
   line 1: Number of Hermite Polynomials:6
   line 2: Number of channels: 2
   line 3: Number of samples: 650,000
   From line 4 til end: approximate center of the heartbeats
   * @param *heartbeatCenters array to save the centers of the heartbeats
   * @param *frequency_hb: pointer to int to save the frequency_hb from the file
   * @param *polynomials: pointer to int to save the number of hermite polynomials
   * @param *channels: pointer to int to save the number of channels
   * @param *polynomials: pointer to int to save the number of total signals (samples)
   * @return: the number of heartbeats read in the input file
*/
int loadHeartbeatcenters(int *heartbeatCenters,float *frequency_hb,int *polynomials, int *channels, int *numSamples)
{
    FILE *heartbeatCentersFile = fopen("/home/user/CUDA/ECG_original/original_Algoritmo_C_Beat_SigmaI_144x1/prueba_Entry_10beats","r");

    if (heartbeatCentersFile==NULL)
    {
        printf("Error opening the file of centers");
        return -1;
    }
    fscanf(heartbeatCentersFile,"%f", frequency_hb);

    fscanf(heartbeatCentersFile,"%i", polynomials);

    fscanf(heartbeatCentersFile,"%i", &channels);

    fscanf(heartbeatCentersFile,"%i", &numSamples);

    int numHeartbeats=0;
    int indexHeartbeats=0; // Read the heartbeat index in the file
    int heartbeatType=0;
    while (!feof(heartbeatCentersFile))
    {
        fscanf (heartbeatCentersFile, "%d,%d,%d\n", &indexHeartbeats,&heartbeatType, &heartbeatCenters[numHeartbeats]);
        numHeartbeats++;
    }
    return numHeartbeats;
}

/** 
   Function: loadsignals
   * Reads the file where the signals are stored and saves all of them in the 
   arrays *signalsChannel1 and *signalsChannel2
   * @param *signalsChannel1 array to save the measurements of the channel 1
   * @param *signalsChannel2 array to save the measurements of the channel 2
*/
void loadsignals(int* signalsChannel1,int* signalsChannel2)
{
    FILE *signalsFile = fopen("/home/user/CUDA/Algoritmo_C_Beat_SigmaI_144x1/prueba_Signal","r");
    if (signalsFile==NULL)
    {
        printf("Error al abrir fichero");
    }
    int signalIndex;
    int i=0;
    int signalChannel1;
    int signalChannel2;
    /*
    Read all the signals, relevant and not relevant
    */
    while (!feof(signalsFile))
    {
        fscanf (signalsFile, "%d,%d,%d\n", &signalIndex,&signalChannel1,&signalChannel2);
        signalsChannel1[i]=signalChannel1;
        signalsChannel2[i]=signalChannel2;
        i++;
    }
}

/** 
   Function: writeHermiteResults
   * Write the heartbeat approximation calculated by the HermitePolynomialsApproximation 
   in the file results
   * @param *hermiteApproximation: array where the approximated heartbeat is in memory. 
   It will be written in the file results
   * @param *resultsFile: pointer to the file where the results are saved
   * @param numHeartbeats: num of heartbeats
*/
void writeHermiteResults(float *hermiteApproximation,FILE *resultsFile,int numHeartbeats)
{
    int i=0;
    int heartbeat=0;
    int indexHeartbeat=0;
    int c;
    char car;
    int bufferPos=0;
    long intPart,decimalPart;
    float mul10=1;
    int precision=9;
    char pBuffer[300];
    char bufHeartbeat[6];

    while(heartbeat<numHeartbeats)
    {
        fputs("\n heartbeat number: ",resultsFile);
        sprintf(bufHeartbeat,"%d",heartbeat);
        fputs(bufHeartbeat,resultsFile);
        fputs("\n aproximacion:",resultsFile);
        indexHeartbeat=heartbeat*NUMSAMPLESHEARTBEAT;
        for(i=0;i<NUMSAMPLESHEARTBEAT;i++)
        {
            bufferPos=0;
            mul10=1;
            for(c=0;c<precision;c++)
            {
                mul10*=10;
            }
            intPart=abs((long)hermiteApproximation[indexHeartbeat+i]);
            decimalPart=abs((long)((hermiteApproximation[indexHeartbeat+i]-(long)hermiteApproximation[indexHeartbeat+i])*mul10));
            do
            {
                pBuffer[bufferPos]=(char)(intPart%10)+'0'; bufferPos++;
                intPart/=10;
            }
            while(intPart>0);
            if(hermiteApproximation[indexHeartbeat+i]<0)
            {
                pBuffer[bufferPos]='-'; bufferPos++;
            }
            for(c=0;c<bufferPos/2;c++)
            {
                car=pBuffer[c]; pBuffer[c]=pBuffer[bufferPos-c-1]; pBuffer[bufferPos-c-1]=car;
            }
            if(precision>0)
            {
                pBuffer[bufferPos]='.'; bufferPos++;
                int parteDecimalPos=bufferPos;
                for(c=0;c<precision;c++)
                {
                   pBuffer[bufferPos]=(char)(decimalPart%10)+'0'; bufferPos++;
                   decimalPart/=10;
               }
               for(c=0;c<precision/2;c++)
               {
                   car=pBuffer[c+parteDecimalPos];
                   pBuffer[c+parteDecimalPos]=pBuffer[bufferPos-c-1];
                   pBuffer[bufferPos-c-1]=car;
               }
           }
           pBuffer[bufferPos]=0;
           fputs(pBuffer,resultsFile);
           fputs(",",resultsFile);
        }  // END SAMPLE HEARTBEATS LOOP
        heartbeat++;
    }  // END HEARTBEATS LOOP
}

/**  
   Function: writeWeightsResult
   * Writes the weights and the best sigma for the heartbeat approximation calculated by the HermitePolynomialsApproximation 
   in the file resultsFile
   * @param *weightshermite: array that contains the hermite weights
   * @param *sigmaBest: array that contains the best sigma for the corresponding hermite weight
   * @param *resultsFile: pointer to the file where the results are saved
   * @param numHeartbeats: num of heartbeats
   * @param polynomials: num of polynomials
*/
void writeWeightsResult(float* weightshermite,float* sigmaBest,FILE *resultsFile,int numHeartbeats,int polynomials)
{
    int i=0;
    int heartbeat=0;
    int indexHeartbeat=0;
    int c;
    char car;
    int bufferPos=0;
    long intPart,decimalPart;
    float mul10=1;
    int precision=9;
    char pBuffer[300];
    char bufHeartbeat[6];
    char bufsigma[6];
    char bufpolynomial[2];
    while(heartbeat<numHeartbeats)
    {
        fputs("\n heartbeat number: ",resultsFile);
        sprintf(bufHeartbeat,"%d",heartbeat);
        fputs(bufHeartbeat,resultsFile);
        fputs(" best sigma: ",resultsFile);
        sprintf(bufsigma,"%f",sigmaBest[heartbeat]);
        fputs(bufsigma,resultsFile);
        indexHeartbeat=heartbeat*polynomials;
        for(i=0;i<polynomials;i++)
        {
            fputs("\n polynomial number: ",resultsFile);
            sprintf(bufpolynomial,"%d",i);
            fputs(bufpolynomial,resultsFile);
            fputs("\t ",resultsFile);
            bufferPos=0;
            mul10=1;
            for(c=0;c<precision;c++)
            {
                mul10*=10;
            }
            intPart=abs((long)weightshermite[indexHeartbeat+i]);
            decimalPart=abs((long)((weightshermite[indexHeartbeat+i]-(long)weightshermite[indexHeartbeat+i])*mul10));
            do
            {
                pBuffer[bufferPos]=(char)(intPart%10)+'0'; bufferPos++;
                intPart/=10;
            }
            while(intPart>0);
            if(weightshermite[indexHeartbeat+i]<0)
            {
                pBuffer[bufferPos]='-'; bufferPos++;
            }
            for(c=0;c<bufferPos/2;c++)
            {
                car=pBuffer[c]; pBuffer[c]=pBuffer[bufferPos-c-1]; pBuffer[bufferPos-c-1]=car;
            }
            if(precision>0)
            {
                pBuffer[bufferPos]='.'; bufferPos++;
                int parteDecimalPos=bufferPos;
                for(c=0;c<precision;c++)
                {
                    pBuffer[bufferPos]=(char)(decimalPart%10)+'0'; bufferPos++;
                    decimalPart/=10;
                }
                for(c=0;c<precision/2;c++)
                {
                    car=pBuffer[c+parteDecimalPos];
                    pBuffer[c+parteDecimalPos]=pBuffer[bufferPos-c-1];
                    pBuffer[bufferPos-c-1]=car;
                }
            }
            pBuffer[bufferPos]=0;
            fputs(pBuffer,resultsFile);
        } // Herte polynomyal weights loop end
        heartbeat++;
    }  // Heartbeat loop end
}

/**  
   Function: writeWeightsResultGPU
   * Writes the weights and the best sigma for the heartbeat approximation calculated by the HermitePolynomialsApproximation in the GPU
   in the file resultsFile
   * @param *weightshermite: array that contains the hermite weights
   * @param *sigmaBest: array that contains the best sigma for the corresponding hermite weight
   * @param *resultsFile: pointer to the file where the results are saved
   * @param numHeartbeats: num of heartbeats
   * @param polynomials: num of polynomials
*/
void writeWeightsResultGPU(float* weightshermite,float* sigmaBest, FILE *resultsFile,int numHeartbeats,int polynomials)
{
    int i=0;
    int heartbeat;  // loop over all the heartbeats
    int channel=0;
    int indice=0;  // string to calculate the index of each heartbeat
    int weightsIndex=0;
    int indicesigmas=0;
    int c;
    char car;
    int bufferPos=0;
    long intPart,decimalPart;
    float mul10=1;
    int precision=9;
    char pBuffer[300];  // String to know the weight of each polynomial
    char bufHeartbeat[6];  // string to know the heartbeat where we are
    char bufsigma[6];   // string to know what is the best sigma
    char bufpolynomial[2];  // string to know in what weight of the polynomial we are

    while(channel<CHANNELS)
    {
        fputs("\n  ############### CHANNEL ",resultsFile);
        sprintf(bufHeartbeat,"%d",channel);
        fputs(bufHeartbeat,resultsFile);
        fputs("   #############################",resultsFile);
        weightsIndex=channel*numHeartbeats*polynomials;
        indicesigmas=channel*numHeartbeats;
        heartbeat=0;
        while(heartbeat<numHeartbeats)
        {
            fputs("\n heartbeat number: ",resultsFile);
            sprintf(bufHeartbeat,"%d",heartbeat);
            fputs(bufHeartbeat,resultsFile);
            fputs(" best sigma: ",resultsFile);
            sprintf(bufsigma,"%f",sigmaBest[indicesigmas+heartbeat]);
            fputs(bufsigma,resultsFile);
            indice=heartbeat*polynomials;
            for(i=0;i<polynomials;i++)
            {
                fputs("\n polynomial number: ",resultsFile);
                sprintf(bufpolynomial,"%d",i);
                fputs(bufpolynomial,resultsFile);
                fputs("\t ",resultsFile);
                bufferPos=0;
                mul10=1;
                for(c=0;c<precision;c++)
                {
                    mul10*=10;
                }
                intPart=abs((long)weightshermite[weightsIndex+indice+i]);
                decimalPart=abs((long)((weightshermite[weightsIndex+indice+i]-(long)weightshermite[weightsIndex+indice+i])*mul10));
                do
                {
                    pBuffer[bufferPos]=(char)(intPart%10)+'0'; bufferPos++;
                    intPart/=10;
                }
                while(intPart>0);
                if(weightshermite[weightsIndex+indice+i]<0)
                {
                    pBuffer[bufferPos]='-'; bufferPos++;
                }
                for(c=0;c<bufferPos/2;c++)
                {
                    car=pBuffer[c]; pBuffer[c]=pBuffer[bufferPos-c-1]; pBuffer[bufferPos-c-1]=car;
                }
                if(precision>0)
                {
                    pBuffer[bufferPos]='.'; bufferPos++;
                    int parteDecimalPos=bufferPos;
                    for(c=0;c<precision;c++)
                    {
                        pBuffer[bufferPos]=(char)(decimalPart%10)+'0'; bufferPos++;
                        decimalPart/=10;
                    }
                    for(c=0;c<precision/2;c++)
                    {
                        car=pBuffer[c+parteDecimalPos];
                        pBuffer[c+parteDecimalPos]=pBuffer[bufferPos-c-1];
                        pBuffer[bufferPos-c-1]=car;
                    }
                }
                pBuffer[bufferPos]=0;
                fputs(pBuffer,resultsFile);
            } // Herte polynomyal weights loop end
            heartbeat++;
        }  // Heartbeat loop end
        channel++;
    }  // channels loop end
}

/**  
   * Function: calculateInitial
   Loads the heartbeat based on the center. It adjusts the real center (highest intensity signal) 
   and calculates the initial position of the hearbeat (QRS signals).
   * @param heartbeatCenter Approximated center of the heartbeat
   * @param *signals array that contains all the signals from the channel of interest
   * @return the initial position of the hearbeat (QRS signals).
*/
int calculateInitial(int heartbeatCenter, int* signalsArray)
{
    int heartbeatInitial=heartbeatCenter-36;
    int sumav1=0;
    int i=0;
    float average;
    int max=0;
    int maxPosition=0;
    while(i<72)
    {
        sumav1+=signalsArray[heartbeatInitial+i];
        i++;
    }
    average=sumav1/72;

    i=0;
    while(i<72)
    {
        if((abs(signalsArray[heartbeatInitial+i] - average))>max)
        {
            max=abs(signalsArray[heartbeatInitial+i] - average);
            maxPosition=heartbeatInitial+i;
        }
        i++;
    }
    return maxPosition-35;
}

/** 
   * Function: calculateAllCenters
   Loads the heartbeats based on the center. It adjusts the real center (highest intensity signal) 
   and calculates the initial position of the hearbeats (QRS signals).
   * @param *heartbeatSignals array that contains all the signals from the channel of interest
   * @param *heartbeatCenters approximated centers of the heartbeats
   * @param *newHeartbeatCenters adjusted centers of the heartbeats.
   * @param numHeartbeats num of total heartbeats
*/
void calculateAllCenters(int *heartbeatSignals, int *heartbeatCenters, int *newHeartbeatCenters, int numHeartbeats)
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
            sumav1 += heartbeatSignals[initialHeartbeat + i];
            i++;
        }
        average1 = sumav1 / 72;

        i = 0;
        while (i < 72)
        {
            if ((abs(heartbeatSignals[initialHeartbeat + i] - average1)) > max)
            {
                max = abs(heartbeatSignals[initialHeartbeat + i] - average1);
                newHeartbeatCenters[centerHeartbeat] = (initialHeartbeat + i);
            }
            i++;
        }
        centerHeartbeat++;
    }
}

/** 
   Function: loadHeartbeat
   * Load the heartbeat based on the center and save the relevant signals (QRS) in *heartbeat. 
   It normalizes the base signal in 0, and filters the noise.
   * @param initial 
   * @param *heartbeat array to save the relevant signals
   * @param *heartbeatSignals array that contains all the signals from the channel of interest
*/
void loadHeartbeat(int initial, int *heartbeat, int *heartbeatSignals)
{
    int i = 0;
    int valueInInitIndex = heartbeatSignals[initial];
    int indexHeartbeat = 0; // If we load different heartbeats at the same time, we need an index
    int j = 0; // To take the relevant signals after the real center is adjusted
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
        heartbeat[i + indexHeartbeat] = heartbeatSignals[initial + j] - valueInInitIndex;
        i++;
        j++;
    }
    while (i < 144)
    {
        heartbeat[i + indexHeartbeat] = 0;
        i++;
    }
}

/**  
   Function: loadallHeartBeats
   * Load the heartbeats based on the center and save the relevant signals (QRS) in *heartbeat. 
   It normalizes the base signal in 0, and filters the noise.
   * @param *heartbeatCentersChannel1 centers of heartbeats in channel1
   * @param *heartbeatCentersChannel2 centers of heartbeats in channel2
   * @param *allHeartbeats array to save the relevant signals from both channels
   * @param *heartbeatSignalsChannel1 signals of heartbeats in channel1
   * @param *heartbeatSignalsChannel2 signals of heartbeats in channel2
   * @param numHeartbeats num of heartbeats in each channel
*/
void loadAllHeartbeats(int *heartbeatCentersChannel1,int *heartbeatCentersChannel2, 
    int *allHeartbeats,int *heartbeatSignalsChannel1,int *heartbeatSignalsChannel2, int numHeartbeats)
{
    int heartbeat=0;
    int indexHeartbeat; // calculate the index for all heartbeats
    int i=0;
    int initial=0;
    int valueInInitIndex=0;
    int j=0; // para coger los valores del array adecuados

    while(heartbeat<numHeartbeats)
    {
        initial=heartbeatCentersChannel1[heartbeat] - 35;
        i=0;
        j=0;
        valueInInitIndex=heartbeatSignalsChannel1[initial];
        indexHeartbeat=heartbeat*144;
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
            allHeartbeats[indexHeartbeat+i]=0;
            i++;
        }
        while(i<108)
        {
            allHeartbeats[indexHeartbeat+i]=heartbeatSignalsChannel1[initial+j]-valueInInitIndex;
            i++;
            j++;
        }
        while(i<144)
        {
            allHeartbeats[indexHeartbeat+i]=0;
            i++;
        }
        heartbeat++;
    } // END LOOP HEARTBEATS


    // START THE READING OF THE SECOND CHANNEL

    heartbeat=0;
    while(heartbeat<numHeartbeats)
    {
        initial=heartbeatCentersChannel2[heartbeat] - 35;
        i=0;
        j=0;
        valueInInitIndex=heartbeatSignalsChannel2[initial];
        indexHeartbeat = (numHeartbeats*144) + (heartbeat*144);
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
            allHeartbeats[i+indexHeartbeat]=0;
            i++;
        }
        while(i<108)
        {
            allHeartbeats[i+indexHeartbeat]=heartbeatSignalsChannel2[initial+j]-valueInInitIndex;
            i++;
            j++;
        }
        while(i<144)
        {
            allHeartbeats[i+indexHeartbeat]=0;
            i++;
        }
        heartbeat++;
    }
}

/** 
   Function: calculateInitialNormalized
   * Calculates the initial position of the heartbeat using the absolute values
   * @param *heartbeat array to save the relevant signals
   * @param *signals array that contains all the signals from the channel of interest
   * @returns the initial position of the heartbeat
*/
int calculateInitialNormalized(int heartbeatCenter, int *heartbeatSignals)
{
    int heartbeatInitial=heartbeatCenter-36;
    int i=0;
    int max=0;
    int maxPosition=0;
    i=0;
    while(i<72)
    {
        if(abs(heartbeatSignals[heartbeatInitial+i])>max)
        {
            max=abs(heartbeatSignals[heartbeatInitial+i]);
            maxPosition=heartbeatInitial+i;
        }
        i++;
    }
    return maxPosition-35;
}

/* 
   Function: calculateAllCentersNormalized
   * Calculates the center position of the heartbeats using the absolute values
   * @param *signals: the array of all signals taken in the corresponding channel
   * @param *hertbeatCenters: array of the approximated centers
   * @param *newHeartbeatCenters array to save the adjusted centers
   * @param numHeartbeats: number of heartbeats
*/
void calculateAllCentersNormalized(int* heartbeatSignals, int* hertbeatCenters,int* newHeartbeatCenters, int numHeartbeats)
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
            if(abs(heartbeatSignals[heartbeat+i])>max)
            {
                max=abs(heartbeatSignals[heartbeat+i]);
                newHeartbeatCenters[heartbeatCenter]=(heartbeat+i);
            }
            i++;
        }
        heartbeatCenter++;
    }
}

/**  
   Function: loadHeartbeatloadHeartbeatNormalized
   * Loads the heartbeat based on the center and saves the relevant signals (QRS) in *heartbeat. 
   It normalizes the base signal in 0, and filters the noise.
   * @param initial 
   * @param *heartbeat array to save the relevant signals
   * @param *signals array that contains all the signals from the channel of interest
*/
void loadHeartbeatNormalized(int initial,int *heartbeat,int *signalsChannel)
{
    int i=0;
    int indexHeartbeat=0; 
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
        heartbeat[i+indexHeartbeat]=0;
        i++;
    }
    while(i<108)
    {
        heartbeat[i+indexHeartbeat]=signalsChannel[initial+j];
        i++;
        j++;
    }
    while(i<144)
    {
        heartbeat[i+indexHeartbeat]=0;
        i++;
    }
}

/** 
   Function: loadAllHeartbeatsNormalized
   * Loads the heartbeats based on the center and saves the relevant signals (QRS) in *heartbeat. 
   It normalizes the base signal in 0, and filters the noise.
   * @param *heartbeatCentersChannel1 centers of heartbeats in channel1
   * @param *heartbeatCentersChannel2 centers of heartbeats in channel2
   * @param *allHeartbeats array to save the relevant signals from both channels
   * @param *channel1 signals of heartbeats in channel1
   * @param *channel2 signals of heartbeats in channel2
   * @param numHeartbeats num of heartbeats in each channel
*/
void loadAllHeartbeatsNormalized(int *heartbeatCentersChannel1,int *heartbeatCentersChannel2,
    int *heartbeats,int *signalsChannel1,int *signalsChannel2,int numHeartbeats)
{
    int heartbeat=0;
    int indexHeartbeat=0; // para hallar el indice y copiar todos los heartbeats
    int i=0;
    int initial=0;
    int j=0; // para coger los valores del array adecuados
    while(heartbeat<numHeartbeats)
    {
        initial=heartbeatCentersChannel1[heartbeat] - 35;
        i=0;
        j=0;
        indexHeartbeat=heartbeat*144;
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
            heartbeats[i+indexHeartbeat]=0;
            i++;
        }
        while(i<108)
        {
            heartbeats[i+indexHeartbeat]=signalsChannel1[initial+j];
            i++;
            j++;
        }
        while(i<144)
        {
            heartbeats[i+indexHeartbeat]=0;
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
        indexHeartbeat=(numHeartbeats*144) + (heartbeat*144);
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
            heartbeats[i+indexHeartbeat]=0;
            i++;
        }
        while(i<108)
        {
            heartbeats[i+indexHeartbeat]=signalsChannel2[initial+j];
            i++;
            j++;
        }
        while(i<144)
        {
            heartbeats[i+indexHeartbeat]=0;
            i++;
        }
        heartbeat++;
    }
}

/** 
   Function: errorMax
   * Calculates the distincts errors () between the hermite approximation by the CPU and the GPU
   and writes them in the file errorsFile.
   * @param *weights1CPU: weights of hermite polynomials for signal recorded in channel1
   * @param *weights2CPU: weights of hermite polynomials for signal recorded in channel2
   * @param *weightsGPU: weights of hermite polynomials for all signals calculated by the GPU
   * @param *sigma1CPU: Best Sigmas for hermite polynomials in channel1
   * @param *sigma2CPU: Best Sigmas for hermite polynomials in channel2
   * @param *sigmaGPU: Best Sigmas for hermite polynomials calculated by GPU in both channels
   * @param numHeartbeats: number of heartbeats
   * @param polynomials: number of polynomials
   * @param errorsFile: File to write the errors
*/
float errorMax(float *weights1CPU,float *weights2CPU, float *weightsGPU, 
    float *sigma1CPU, float *sigma2CPU, float *sigmaGPU, int numHeartbeats, int polynomials, FILE *errorsFile)
{
    float errorMax=0.0;
    float sigmaError=0.0;
    float averageValue[polynomials];
    float relativeError[polynomials];
    float averageError[polynomials];
    int polynomial=0;
    int heartbeat=0;
    int sampleHeartbeatsIndex=0;
    while(polynomial<polynomials)
    {
        averageValue[polynomial]=0.0;
        averageError[polynomial]=0.0;
        relativeError[polynomial]=0.0;
        polynomial++;
    }
    while(heartbeat<numHeartbeats)
    {
        polynomial=0;
        sampleHeartbeatsIndex=heartbeat*polynomials;
        while(polynomial<polynomials)
        {
            if(abs(weights1CPU[sampleHeartbeatsIndex+polynomial]-weightsGPU[sampleHeartbeatsIndex+polynomial])>errorMax)
            {
                errorMax=abs(weights1CPU[sampleHeartbeatsIndex+polynomial]-weightsGPU[sampleHeartbeatsIndex+polynomial]);
            }
            averageValue[polynomial]+=weightsGPU[sampleHeartbeatsIndex+polynomial];
            averageError[polynomial]+=pow(abs(weights1CPU[sampleHeartbeatsIndex+polynomial]-weightsGPU[sampleHeartbeatsIndex+polynomial]),2);
            polynomial++;
        }
        if(abs(sigma1CPU[heartbeat]-sigmaGPU[heartbeat])>sigmaError)
        {
            sigmaError=abs(sigma1CPU[heartbeat]-sigmaGPU[heartbeat]);
        }
        heartbeat++;
    } // END HEARTBEATS LOOP

    heartbeat=0;
    int channelIndex=numHeartbeats*polynomials;
    while(heartbeat<numHeartbeats)
    {
        polynomial=0;
        sampleHeartbeatsIndex=heartbeat*polynomials;
        while(polynomial<polynomials)
        {
            if(abs(weights2CPU[sampleHeartbeatsIndex+polynomial]-weightsGPU[channelIndex+sampleHeartbeatsIndex+polynomial])>errorMax)
            {
                errorMax=abs(weights2CPU[sampleHeartbeatsIndex+polynomial]-weightsGPU[channelIndex+sampleHeartbeatsIndex+polynomial]);
            }
            averageValue[polynomial]+=weightsGPU[channelIndex+sampleHeartbeatsIndex+polynomial];
            averageError[polynomial]+=pow(abs(weights2CPU[sampleHeartbeatsIndex+polynomial]-weightsGPU[channelIndex+sampleHeartbeatsIndex+polynomial]),2);
            polynomial++;
        }
        if(abs(sigma2CPU[heartbeat]-sigmaGPU[numHeartbeats + heartbeat])>sigmaError)
        {
            sigmaError=abs(sigma2CPU[heartbeat]-sigmaGPU[numHeartbeats+heartbeat]);
        }
        heartbeat++;
    } // END HEARTBEATS LOOP

    polynomial=0;
    while(polynomial<polynomials)
    {
        averageError[polynomial]=sqrt(averageError[polynomial]/numHeartbeats);
        averageValue[polynomial]=averageValue[polynomial]/numHeartbeats;
        relativeError[polynomial]=abs((averageError[polynomial]/averageValue[polynomial])*100);
        polynomial++;
    }

    int c;
    char car;
    int bufferPos=0;
    long intPart,decimalPart;
    float mul10=1;
    int precision=9;
    char pBuffer[300];

    fputs("\n Error sigma: ",errorsFile);
    fputs("\t ",errorsFile);

    for(c=0;c<precision;c++)
    {
        mul10*=10;
    }
    intPart=abs((long)sigmaError);
    decimalPart=abs((long)((sigmaError-(long)sigmaError)*mul10));
    do
    {
        pBuffer[bufferPos]=(char)(intPart%10)+'0'; bufferPos++;
        intPart/=10;
    }
    while(intPart>0);
    if(sigmaError<0)
    {
        pBuffer[bufferPos]='-'; bufferPos++;
    }
    for(c=0;c<bufferPos/2;c++)
    {
        car=pBuffer[c]; pBuffer[c]=pBuffer[bufferPos-c-1]; pBuffer[bufferPos-c-1]=car;
    }
    if(precision>0)
    {
        pBuffer[bufferPos]='.'; bufferPos++;
        int decimalPartPos=bufferPos;
        for(c=0;c<precision;c++)
        {
            pBuffer[bufferPos]=(char)(decimalPart%10)+'0'; bufferPos++;
            decimalPart/=10;
        }
        for(c=0;c<precision/2;c++)
        {
            car=pBuffer[c+decimalPartPos];
            pBuffer[c+decimalPartPos]=pBuffer[bufferPos-c-1];
            pBuffer[bufferPos-c-1]=car;
        }
    }
    pBuffer[bufferPos]=0;
    fputs(pBuffer,errorsFile);

    bufferPos=0;
    mul10=1;
    fputs("\n Maximum error: ",errorsFile);
    fputs("\t ",errorsFile);

    for(c=0;c<precision;c++)
    {
        mul10*=10;
    }
    intPart=abs((long)errorMax);
    decimalPart=abs((long)((errorMax-(long)errorMax)*mul10));
    do
    {
        pBuffer[bufferPos]=(char)(intPart%10)+'0'; bufferPos++;
        intPart/=10;
    }
    while(intPart>0);
    if(errorMax<0)
    {
        pBuffer[bufferPos]='-'; bufferPos++;
    }
    for(c=0;c<bufferPos/2;c++)
    {
        car=pBuffer[c]; pBuffer[c]=pBuffer[bufferPos-c-1]; pBuffer[bufferPos-c-1]=car;
    }
    if(precision>0)
    {
        pBuffer[bufferPos]='.'; bufferPos++;
        int decimalPartPos=bufferPos;
        for(c=0;c<precision;c++)
        {
            pBuffer[bufferPos]=(char)(decimalPart%10)+'0'; bufferPos++;
            decimalPart/=10;
        }
        for(c=0;c<precision/2;c++)
        {
            car=pBuffer[c+decimalPartPos];
            pBuffer[c+decimalPartPos]=pBuffer[bufferPos-c-1];
            pBuffer[bufferPos-c-1]=car;
        }
    }
    pBuffer[bufferPos]=0;
    fputs(pBuffer,errorsFile);

    fputs("\n Quadratic error average: ", errorsFile);
    fputs("\t ", errorsFile);
    polynomial=0;
    char bufpolynomial[6];
    while(polynomial<polynomials)
    {   
        bufferPos=0;
        mul10=1;
        fputs("\n polynomial number: ", errorsFile);
        sprintf(bufpolynomial,"%d",polynomial);
        fputs(bufpolynomial, errorsFile);
        fputs("\t ", errorsFile);
        for(c=0;c<precision;c++)
        {
            mul10*=10;
        }
        intPart=abs((long)averageError[polynomial]);
        decimalPart=abs((long)((averageError[polynomial]-(long)averageError[polynomial])*mul10));
        do
        {
            pBuffer[bufferPos]=(char)(intPart%10)+'0'; bufferPos++;
            intPart/=10;
        }
        while(intPart>0);
        if(averageError[polynomial]<0)
        {
            pBuffer[bufferPos]='-'; bufferPos++;
        }
        for(c=0;c<bufferPos/2;c++)
        {
            car=pBuffer[c]; pBuffer[c]=pBuffer[bufferPos-c-1]; pBuffer[bufferPos-c-1]=car;
        }
        if(precision>0)
        {
            pBuffer[bufferPos]='.'; bufferPos++;
            int decimalPartPos=bufferPos;
            for(c=0;c<precision;c++)
            {
                pBuffer[bufferPos]=(char)(decimalPart%10)+'0'; bufferPos++;
                decimalPart/=10;
            }
            for(c=0;c<precision/2;c++)
            {
                car=pBuffer[c+decimalPartPos];
                pBuffer[c+decimalPartPos]=pBuffer[bufferPos-c-1];
                pBuffer[bufferPos-c-1]=car;
            }
        }
        pBuffer[bufferPos]=0;
        fputs(pBuffer, errorsFile);
        polynomial++;
    }

    fputs("\n Average Value: ", errorsFile);
    fputs("\t ", errorsFile);
    polynomial=0;

    while(polynomial<polynomials)
    {   
        bufferPos=0;
        mul10=1;
        fputs("\n coeficiente numero: ", errorsFile);
        sprintf(bufpolynomial,"%d",polynomial);
        fputs(bufpolynomial, errorsFile);
        fputs("\t ", errorsFile);
        for(c=0;c<precision;c++)
        {
            mul10*=10;
        }
        intPart=abs((long)averageValue[polynomial]);
        decimalPart=abs((long)((averageValue[polynomial]-(long)averageValue[polynomial])*mul10));
        do
        {
            pBuffer[bufferPos]=(char)(intPart%10)+'0'; bufferPos++;
            intPart/=10;
        }
        while(intPart>0);
        if(averageValue[polynomial]<0)
        {
            pBuffer[bufferPos]='-'; bufferPos++;
        }
        for(c=0;c<bufferPos/2;c++)
        {
            car=pBuffer[c]; pBuffer[c]=pBuffer[bufferPos-c-1]; pBuffer[bufferPos-c-1]=car;
        }
        if(precision>0)
        {
            pBuffer[bufferPos]='.'; bufferPos++;
            int decimalPartPos=bufferPos;
            for(c=0;c<precision;c++)
            {
                pBuffer[bufferPos]=(char)(decimalPart%10)+'0'; bufferPos++;
                decimalPart/=10;
            }
            for(c=0;c<precision/2;c++)
            {
                car=pBuffer[c+decimalPartPos];
                pBuffer[c+decimalPartPos]=pBuffer[bufferPos-c-1];
                pBuffer[bufferPos-c-1]=car;
            }
        }
        pBuffer[bufferPos]=0;
        fputs(pBuffer, errorsFile);
        polynomial++;
    }

    fputs("\n Average Relative error en %: ", errorsFile);
    fputs("\t ", errorsFile);
    polynomial=0;
    while(polynomial<polynomials)
    {   
        bufferPos=0;
        mul10=1;
        fputs("\n polynomial number: ", errorsFile);
        sprintf(bufpolynomial,"%d",polynomial);
        fputs(bufpolynomial, errorsFile);
        fputs("\t ", errorsFile);
        for(c=0;c<precision;c++)
        {
            mul10*=10;
        }
        intPart=abs((long)relativeError[polynomial]);
        decimalPart=abs((long)((relativeError[polynomial]-(long)relativeError[polynomial])*mul10));
        do
        {
            pBuffer[bufferPos]=(char)(intPart%10)+'0'; bufferPos++;
            intPart/=10;
        }
        while(intPart>0);
        if(relativeError[polynomial]<0)
        {
            pBuffer[bufferPos]='-'; bufferPos++;
        }
        for(c=0;c<bufferPos/2;c++)
        {
            car=pBuffer[c]; pBuffer[c]=pBuffer[bufferPos-c-1]; pBuffer[bufferPos-c-1]=car;
        }
        if(precision>0)
        {
            pBuffer[bufferPos]='.'; bufferPos++;
            int decimalPartPos=bufferPos;
            for(c=0;c<precision;c++)
            {
                pBuffer[bufferPos]=(char)(decimalPart%10)+'0'; bufferPos++;
                decimalPart/=10;
            }
            for(c=0;c<precision/2;c++)
            {
                car=pBuffer[c+decimalPartPos];
                pBuffer[c+decimalPartPos]=pBuffer[bufferPos-c-1];
                pBuffer[bufferPos-c-1]=car;
            }
        }
        pBuffer[bufferPos]=0;
        fputs(pBuffer, errorsFile);
        polynomial++;
    }
    return errorMax;
}
