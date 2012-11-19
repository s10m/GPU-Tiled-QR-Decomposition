/**
 * @file qrdecomp.c
 * @author Sam Townsend
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "qrdecomp.h"

#define CO(i,j,m) ((m * j) + i)
#define TIMES 1

int main(int argc, char* argv[])
{
	double* matQ = NULL, *matA = NULL;
	int mq = 5, nq = mq, ma = nq, na = ma, b = mq;

	srand(1);

	matQ = newMatrix(matQ, mq, nq);
	matA = newMatrix(matA, ma, na);

	initMatrix(matQ, mq, nq, 1);
	initMatrix(matA, ma, na, 1);

	printMatrix(matQ, mq, nq);
	printf("\n");
	//printMatrix(matA, ma, na);

	//matC = multAB(matQ, mq, nq, matA, ma, na,0,0);

	//printMatrix(matC, mq, na);
	hhQR(matA, b, b, b, mq);
	/*hhQR(matQ, b, b, b, 1, 1, mq);
	hhQR(matQ, b, b, b, 2, 2, mq);
	hhQR(matQ, b, b, b, 3, 3, mq);*/
	//printf("R =\n");
	printMatrix(matQ, mq, nq);
	deleteMatrix(matQ);
	deleteMatrix(matA);
	//deleteMatrix(matC);
	return 0;
}
/**
 * \breif Computes the \f$QR\f$ decomposition of the \f$b\times b\f$ portion of a matrix stored at matA
 *
 * 
 * 
 * \param matA A matrix in column-major order as a pointer to an array
 * \param m The number of rows in the matrix as stored
 * \param n The number of columns in the matrix as stored
 * \param b The block size to compute
 * \param roffset The row-offset into the matrix to start from
 * \param coffset The column-offset into the matrix to start from
 * \param lda The number of rows of the matrix as stored in memory
 * \returns A freshly allocated pointer to a matrix of pointers containing the householder reflectors from the decomposition.
 */
double** hhQR(double* matA, int m, int n, int b, int lda)//returns Q and triangularises matA in place.
{
	int k;
	double* xVect, **v = malloc(b * sizeof(double*));
	double* matQ = NULL, *matAgain;
	matQ = newMatrix(matQ, b, b);
	initMatrix(matQ, b, b, 2);

	for(k = 0; k < b; k++)
	{
		//x = matA(k:m,k)
		xVect = matA + CO(k,k,lda);//xVect is column vector from k -> b-k in column k of block

		//vk = sign(x[1])||x||_2e1 + x
		//vk = vk/||vk||_2
		v[k] = calcvk(xVect, b - k);

		//matA(k:ma,k:na) = matA(k:ma,k:na) - 2((vk*vk.T)/(vk.T*vk))*matA(k:ma,k:na)
		updateMatHHQRInPlace(matA, k, m, n, b, lda, v[k], b - k);
	}

	for(k = b - 1; k > -1; k --)
	{
		updateMatHHQRInPlace(matQ, k, b, b, b, b, v[k], b - k);
		free(v[k]);
	}
	printMatrix(matQ, b, b);

	matAgain = multAB(matQ, b, b, matA, b, b);
	printMatrix(matAgain, b, b);
	
	free(v);
	return v;
}
/**
 * \brief Applies the householder update to the matrix
 * 
 * Computes the update: \f[a_{ij} = a_{ij} + \left(v_i\frac{-2}{\sum_{k=1}^mv_k^2}\right)\left(\sum_{k=1}^ma_{kj}v_k\right)\f]
 * 
 * \param mat Matrix of size \f$m \times n\f$
 * \param koffset The offset diagonally into the block, defining the subblock \f$A^{\prime}(k:b,k:b)\f$
 * \param m The number of rows in the matrix as stored
 * \param n The number of columns in the matrix as stored
 * \param b The block size in the matrix
 * \param v Pointer to an array containing the Householder reflector \f$v\f$
 * \param l The number of elements in \f$v\f$
 */

void updateMatHHQRInPlace(
		double* mat,
		int koffset,
		int m,
		int n,
		int b,
		int ldm,
		double* v,
		int l)
{
	int i, j, k, rows = m-koffset, cols = n-koffset;
	double y = 0, z, a;

	//Calculate y := -2/(sum(v[k]^2)) (lines 2-5 in Algorithm 1)
	for(k = 0; k < rows; k ++)
		y += (v[k]*v[k]);
	//y=1;
	y = -2/y;

	for(j = 0; j < cols; j ++)
	{
		//calculate z := sum(a(k,j) * v[k]) (lines 7 - 10 in Algorithm 1)
		z = 0;
		for(k = 0; k < rows; k ++)
			z += mat[CO((k+koffset),(j+koffset),ldm)] * v[k];

		//apply A(i,j) := A(i,j) + v[i] * y * z (lines 11 - 15 in Algorithm 1)
		for(i = 0; i < cols; i ++)
		{
			a = y * z;
			a *= v[i];
			mat[CO((i+koffset),(j+koffset),ldm)] += a;
		}
	}
}

/**
 * \brief Computes a Householder reflector \f$v\f$ of a vector \f$x\f$
 *
 * Computes: \f[v = \textrm{sign}(x_1)||x||_2e_1+x\f]
 * Then normalises \f$v\f$: \f[v = \frac{v}{||v||_2}\f]
 * 
 * \param x Pointer to an array containing a column vector to compute the Householder reflector of
 * \param l The number of elements in \f$x\f$
 *
 * \returns A pointer to a newly allocated vector \f$v\f$, a Householder reflector of \f$x\f$
 */
double* calcvk(double* x, int l)
{
	int sign, i;
	double vk2norm, toadd, *vk;
	double div;

	vk = malloc(l * sizeof(double));

	for(i = 0; i < l; i++)
		vk[i] = x[i];

	sign = vk[0] >= 0.0 ? 1 : -1;

	vk2norm = do2norm(vk, l);
	toadd = sign * vk2norm;
	div = 1/(x[0]+toadd);
	
	for(i = 1; i < l; i++)
		vk[i] *= div;

	vk[0] = 1;

	return vk;
}

/**
 * \brief Normalises a vector \f$v\f$
 *
 * computes and returns the normalised vector by multiplication: \f[v\frac{1}{||v||}\f]
 * \param v Pointer to an array containing the vector to be normalised
 * \param l The number of elements in the vector \f$v\f$
 * \param norm The pre-computed norm of the vector
 * \returns The first parameter, the vector \f$v\f$
 */
double* normalisev(double* v, int l, double norm)
{
	int i;
	double divNorm = 1.0/norm;
	
	for(i = 0; i < l; i++)
		v[i] *= divNorm;
	
	return v;
}

/*
  \brief Computes 2-norm of a vector \f$x\f$
  
  Computes the 2-norm by computing the following: \f[\textrm{2-norm}=\sqrt_0^lx(i)^2\f]
 */
double do2norm(double* x, int l)
{
	double sum = 0, norm;
	int i;

	for(i = 0; i < l; i++)
		sum += x[i] * x[i];

	norm = sqrt(sum);

	return norm;
}

double* multAB(double* matQ, int mq, int nq, double* matA, int ma, int na)
{
	double* matC = NULL;
	int i, j, k;
	matC = newMatrix(matC, mq, na);
	initMatrix(matC, mq, na, 0);

	for(j = 0; j < na; j++)
	{
		for(k = 0; k < nq; k++)
		{
			for(i = 0; i < mq; i++)
			{
				matC[CO(i,j,mq)] += matQ[CO((i),(k),mq)] * matA[CO((k),(j),ma)];
			}
		}
	}

	return matC;
}

void printMatrix(double* mat, int m, int n)
{
	int r, c;
	putchar('{');
	for(r = 0; r < m; r++)
	{
		putchar('{');
		for(c = 0; c < n; c++)
		{
			printf("%7.2f", mat[CO(r,c,m)]);
			if(c != n-1)
				putchar(',');
		}
		putchar('}');
		if(r != m-1)
			putchar(',');
	}
	printf("}\n");
}

void initMatrix(double* mat, int m, int n, int mode)
{
	int r, c;

	for(c = 0; c < n; c++)
	{
		for(r = 0; r < m; r++)
		{
			if(mode == 0)
				mat[CO(r,c,m)] = 0;
			else if(mode == 1)
				mat[CO(r,c,m)] = rand() % (m * n);
			else if(mode == 2)
				mat[CO(r,c,m)] = r == c ? 1 : 0;
		}
	}
}

void deleteMatrix(double* matptr)
{
	free(matptr);
}

double* newMatrix(double* matptr, int m, int n)
{
	matptr = malloc(m * n * sizeof(double));

	return matptr;
}
