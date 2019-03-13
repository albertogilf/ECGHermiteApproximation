#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hermite.h"

/* Author: Alberto Gil de la Fuente
   Purpose: to provide functions to calculate Hermite Polynomials
   See https://en.wikipedia.org/wiki/Hermite_polynomials for more info about Hermite Polynomials
   Language: C
   mail: alberto.gilf@gmail.com
*/

/* 
   Function: HR 
   Recursive implementation of Hermite Polynomials. 
   @param n is the nth order Hermite polynomial 
   @param x 
   @return: the value of the nth order Hermite Polynomial approximation for the value x
*/

float HR(int n, float x)
{
    switch (n)
    {
        case 0:
        {
            return 1;
        }
        case 1:
        {
            return 2 * x;
        }
        default:
        {
            return (2 * x * HR(n - 1, x)) - (2 * (n - 1) * HR(n - 2, x));
        }
    }
}

/* 
   Function: H 
   Declarative implementation of Hermite Polynomials. The maximum grade it can compute is 10th
   @param n is the nth order Hermite polynomial 
   @param x 
   @return: the value of the nth order Hermite Polynomial approximation for the value x
*/

float H(int n, float x)
{
    float resultado = 0;
    if (n == 0)
    {
        resultado = 1.00;
    }
    else if (n == 1)
    {
        resultado = 2 * x;
    }
    else if (n == 2)
    {
        resultado = 4 * (float)pow(x, 2) - 2;
    }
    else if (n == 3)
    {
        resultado = 8 * (float)pow(x, 3) - 12 * x;
    }
    else if (n == 4)
    {
        resultado = 16 * (float)pow(x, 4) - 48 * (float)pow(x, 2) + 12;
    }
    else if (n == 5)
    {
        resultado = 32 * (float)pow(x, 5) - 160 * (float)pow(x, 3) + 120 * x;
    }
    else if (n == 6)
    {
        resultado = 64 * (float)pow(x, 6) - 480 * (float)pow(x, 4) + 720 * (float)pow(x, 2) - 120;
    }
    else if (n == 7)
    {
        resultado = 128 * (float)pow(x, 7) - 1344 * (float)pow(x, 5) + 3360 * (float)pow(x, 3) - 1680 * x;
    }
    else if (n == 8)
    {
        resultado = 256 * (float)pow(x, 8) - 3584 * (float)pow(x, 6) + 13440 * (float)pow(x, 4) - 13440 * (float)pow(x, 2) + 1680;
    }
    else if (n == 9)
    {
        resultado = 512 * (float)pow(x, 9) - 9216 * (float)pow(x, 7) + 48384 * (float)pow(x, 5) - 80640 * (float)pow(x, 3) + 30240 * x;
    }
    else if (n == 10)
    {
        resultado = 1024 * (float)pow(x, 10) - 23040 * (float)pow(x, 8) + 161280 * (float)pow(x, 6) - 403200 * (float)pow(x, 4) +
                    302400 * (float)pow(x, 2) - 30240;
    }
    else
    {
        resultado = 0;
    }
    return resultado;
}

/**
     * Function: H0
     * Hermite Polynomial of order 0 (n=0)
     * @return 1.00
     */
float H0()
{
    return 1.00;
}

/**
     * Function: H1
     * Hermite Polynomial of order 1 (n=1)
     * @param x index
     * @return 2 * X
     */
float H1(float x)
{
    return 2 * x;
}

/**
     * Function: H2
     * Hermite Polynomial of order 2 (n=2)
     * @param x index
     * @return 4 * pow(x,2) - 2
     */
float H2(float x)
{
    return 4 * (float)pow(x, 2) - 2;
}

/**
     * Function: H3
     * Hermite Polynomial of order 3 (n=3)
     * @param x index
     * @return 8 * pow(x,3) - 12 * x
     */
float H3(float x)
{
    return 8 * (float)pow(x, 3) - 12 * x;
}

/**
     * Function: H4
     * Hermite Polynomial of order 4 (n=4)
     * @param x index
     * @return 16 * pow(x,4) - 48 * pow(x,2) + 12
     */
float H4(float x)
{
    return 16 * (float)pow(x, 4) - 48 * (float)pow(x, 2) + 12;
}

/**
     * Function: H5
     * Hermite Polynomial of order 5 (n=5)
     * @param x index
     * @return 32 * pow(x,5) - 160 * pow(x,3) + 120 * x
     */
float H5(float x)
{
    return 32 * (float)pow(x, 5) - 160 * (float)pow(x, 3) + 120 * x;
}

/**
     * Function: H6
     * Hermite Polynomial of order 6 (n=6)
     * @param x index
     * @return 64 * pow (x,6) - 480 * pow(x,4) + 720 * pow(x,2) - 120
     */
float H6(float x)
{
    return 64 * (float)pow(x, 6) - 480 * (float)pow(x, 4) + 720 * (float)pow(x, 2) - 120;
}

/**
     * Function: H7
     * Hermite Polynomial of order 7 (n=7)
     * @param x index
     * @return 128 * pow(x,7) - 1344 * pow(x,5) + 3360 * pow(x,3) + 1680 * x
     */
float H7(float x)
{
    return 128 * (float)pow(x, 7) - 1344 * (float)pow(x, 5) + 3360 * (float)pow(x, 3) - 1680 * x;
}

/**
     * Function: H8
     * Hermite Polynomial of order 8 (n=8)
     * @param x index
     * @return 256 * pow (x,8) - 3584 * pow (x,6) + 13440 * pow(x,4) - 13440 * pow(x,2) +1680
     */
float H8(float x)
{
    return 256 * (float)pow(x, 8) - 3584 * (float)pow(x, 6) + 13440 * (float)pow(x, 4) - 13440 * (float)pow(x, 2) + 1680;
}

/**
     * Function: H9
     * Hermite Polynomial of order 9 (n=9)
     * @param x index
     * @return 512 * pow(x, 9) - 9216 * pow(x,7) + 48384 * pow(x,5) - 80640 * pow(x,3) + 30240 * x
     */
float H9(float x)
{
    return 512 * (float)pow(x, 9) - 9216 * (float)pow(x, 7) + 48384 * (float)pow(x, 5) - 80640 * (float)pow(x, 3) + 30240 * x;
}

/**
     * Function: H10
     * Hermite Polynomial of order 10 (n=10)
     * @param x index
     * @return 512 * pow(x, 9) - 9216 * pow(x,7) + 48384 * pow(x,5) - 80640 * pow(x,3) + 30240 * x
     */
float H10(float x)
{
    return 1024 * (float)pow(x, 10) - 23040 * (float)pow(x, 8) + 161280 * (float)pow(x, 6) - 403200 * (float)pow(x, 4) +
           302400 * (float)pow(x, 2) - 30240;
}
