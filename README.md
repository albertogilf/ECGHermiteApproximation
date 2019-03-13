# CUDA ECG hermite approximation 

This repo contains the source code to process the QRS signal from ECG using hermite Polynomials. The source code is uploaded to approximate the heartbeats with 6 hermite polynomials, but it has been tested until 28 Hermite Polynomials. 
The current version does not accept the input files as parameters, it is necessary to change them in the source code. 

## Installation
The source code is available in C and a parallel version in CUDA. 
To execute the C Code the compiler gcc should be installed.
To execute the CUDA Code the compiler nvcc should be installed, as well as the CUDA drivers, and a NVIDIA Cuda capable graphic processing unit is needed. Please check the [CUDA developers zone](https://developer.nvidia.com/cuda-zone). 

## Execution
The CUDA version of the code contains a makefile for linux systems. The makefile is set up for CUDA version 10-1. Please consider to modify the CUDA_PATH in the makefile for other versions. 
Check also where you have the common libraries of NVIDIA, although the default path is set to $(CUDA_PATH)/samples/common/inc.

## Experimenting
You can try to see how the speedup changes depending on the number of beats per file used, as well as using different files. I encourage you to play with the Block and Grid dimensions as well. The code presented here uses performance techniques learnt in the CUDA best Practices manual like loop unrolling or Streams. It is also encouraged to play with the code and try to improve the speedup!

## Considerations

* The speedup reached is shown in a screen and saved into a file, as well as the absolute times of the different steps (CPU execution, GPU execution, read files, etc.)
* The sample files provided are called test_signal for the ECG signal, and test_entry, test_entry_10beats, test_entry_100beats, test_entry_1000beats to provide different file sizes. 
The files test_entry_\* contain:
    * Line 1: frequency
    * Line 2: Hermite Polynomials used to approximate the hearbeat
    * Line 3: Number of channels used to measure the ECG
    * Line 4: Number of samples available in the file test_signal
    * From line 5- EOF: \[heartbeat_index\], \[heartbeat_type\], \[approximated_center_of_the_heartbeat\]
The heartbeat type is not used in the hermite approximation, and the approximated center will be used to calculate the real center of the QRS signal.
* If you are interested in the real files used for processing the ECG, please contact with me and/or [David Gonzalez Marquez](https://github.com/DavidGMarquez). 
* This work was the final project of the Bachellor's degree in Computer Science of Alberto Gil de la fuente, in the university CEU-San Pablo, and supervised by [Gabriel Caffarena](https://twitter.com/gacaffe?lang=es). The project memory is available in the library of the university. 
* The source code in Java was developed by [David Gonzalez Marquez](https://github.com/DavidGMarquez). 

[CUDA ECG article](https://github.com/albertogilf/ECGHermiteApproximation/blob/master/gil2016.pdf)