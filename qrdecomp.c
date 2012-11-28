/**
 * @file qrdecomp.c
 * @author Sam Townsend
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "qrdecomp.h"

#define CO(i,j,m) ((m * j) + i)
#define TIMES 1

int main	(int argc,
		char* argv[])
{
	double* matQ = NULL, *matA = NULL, **hhVectors;
	int mq = 5, nq = mq, ma = 4, na = 2, b = 2;

	srand(5);

	matQ = newMatrix(matQ, mq, nq);
	matA = newMatrix(matA, ma, na);

	initMatrix(matQ, mq, nq, 1);
	initMatrix(matA, ma, na, 1);

	printMatrix(matA, ma, na, ma);
	printf("\n");
	//printMatrix(matA, ma, na);

	//matC = multAB(matQ, mq, nq, matA, ma, na,0,0);

	allocVectors(&hhVectors, ma, na);
	
	//printMatrix(matC, mq, na);
	//hhQR(matA, na, na, ma);
	//qRSingleBlock(matA, 2*b,b, ma, hhVectors);
	qRDoubleBlock(matA, b, b, matA + 2, b, ma, hhVectors);
	printMatrix(matA, ma, na, ma);
	//printf("R =\n");
	//printMatrix(matQ, mq, nq);
	deleteMatrix(matQ);
	deleteMatrix(matA);
	//deleteMatrix(matC);
	return 0;
}

void allocVectors	(double*** hhVectors,
			int m,
			int n)
{
	int i;
	
	*hhVectors = malloc(n * sizeof(double*));
	
	for(i = 0; i < n; i++)
	{
		(*hhVectors)[i] = malloc((m - n) * sizeof(double));
	}
}

/**
 * \brief Computes the QR decomposition of a block
 * 
 * Computes the QR decomposition of an \f$m\times n\f$ block in place and stores the householder reflectors
 * in a pre-allocated triangular array, passed as an argument.
 * \param block A pointer to the first element of the block
 * \param m The number of rows in the block
 * \param n The number of columns in the block
 * \param ldb The leading dimension of the matrix
 * \param hhVectors A pointer to a pre-allocated triangular array for storing the n householder vectors
 *
 * \returns void
 */
void qRSingleBlock	(double* block,
			int m,
			int n,
			int ldb,
			double** hhVectors)
{
	int k;
	double* xVect;

	printMatrix(block, m, n, ldb);

	for(k = 0; k < n; k++)
	{
		//x = matA(k:m,k)
		xVect = block + CO(k,k,ldb);//xVect is defo column vector from k -> b-k in column k of block
		//vk = sign(x[1])||x||_2e1 + x
		//vk = vk/||vk||_2
		calcvkSingle(xVect, m - k, hhVectors[k]);

		//matA(k:ma,k:na) = matA(k:ma,k:na) - 2((vk*vk.T)/(vk.T*vk))*matA(k:ma,k:na)
		updateSingleQ(block, k, m, n, ldb, hhVectors[k], m - k);
	}

	printMatrix(block, m, n, ldb);
}

/**
 * \brief Computes the QR decomposition of the matrix formed by coupling two blocks
 * 
 * Computes the QR decomposition of a rectangular block formed by coupling two blocks
 * from within the same matrix on top of each other.
 * Stores the R factor in place and stores the householder reflectors 
 * in a pre-allocated triangular array, passed as an argument.
 *
 * \param blockA A pointer to the first element of the "top" block
 * \param am The number of rows in blockA
 * \param an The number of columns in blockA
 * \param blockB A pointer to the first element of the "bottom" block
 * \param bm The number of rows in blockB
 * \param ldm The leading dimension of the main matrix
 * \param hhVectors A pointer to a pre-allocated triagnular array for storing the n householder vectors
 *
 * \returns void
 */
void qRDoubleBlock	(double* blockA,
			int am,
			int an,
			double* blockB,
			int bm,
			int ldm,
			double** hhVectors)
{
	int k;
	double* xVectB, *xVectA;

	printMatrix(blockA, am, an, ldm);
	printMatrix(blockB, bm, an, ldm);

	for(k = 0; k < an; k++)
	{
		//x = matA(k:m,k)
		xVectA = blockA + CO(k,k,ldm);//xVect is column vector from k -> b-k in column k of block
		xVectB = blockB + CO(0,k,ldm);

		//vk = sign(x[1])||x||_2e1 + x
		//vk = vk/||vk||_2
		calcvkDouble(xVectA, am - k, xVectB, bm, (bm + am) - k, hhVectors[k]);

		//matA(k:ma,k:na) = matA(k:ma,k:na) - 2((vk*vk.T)/(vk.T*vk))*matA(k:ma,k:na)
		//updateMatHHQRInPlace(block, k, m, n, ldb, hhVectors[k], m - k);
		updateDoubleQ(blockA, k, am, an, blockB, bm, ldm, hhVectors[k], (am + bm) - k);//update top block
	}
	
	printMatrix(blockA, am, an, ldm);
	printMatrix(blockB, bm, an, ldm);
}

/** 
 * \brief Applies precomputed householder vectors to a single block within a matrix
 * 
 * Applies the computed householder vectors to a single block of a matrix in place:
 * \f[Q_1Q_2\ldots Q_nM = QM\f]
 *
 * \param block A pointer to the first element of the block to apply the vectors to
 * \param m The number of rows in the block
 * \param n The number of columns in the block
 * \param ldb The leading dimension of the main matrix
 * \param hhVectors A pointer to a triangular array containing n householder vectors 
 */
void applySingleBlock	(double* block,
			int m,
			int n,
			int ldb,
			double** hhVectors)
{
	int k;

	for(k = n - 1; k > 0; k --)
		updateSingleQ(block, k, m, n, ldb, hhVectors[k], m - k);
}

/**
 * \brief Applies precomputed householder vectors to a rectangular block formed from two blocks within the same matrix
 * 
 * Applies the n precomputed householder vectors to the rectangular matrix
 * formed by coupling blockA on top of blockB.
 * \param blockA A pointer to the first element of the "top" block in the coupling
 * \param am The number of rows in blockA
 * \param blockB A pointer to the first element of the "bottom" block in the coupling
 * \param bm The number of rows in the "bottom" block
 * \param n The number of columns in the coupling
 * \param ldm The leading dimension of the matrix where both blocks reside
 * \param hhVectors A pointer to a triangular array containing n householder vectors
 */
void applyDoubleBlock	(double* blockA,
			int am,
			double* blockB,
			int bm,
			int n,
			int ldm,
			double** hhVectors)
{
	int k;

	for(k = n - 1; k > 0; k --)
		updateDoubleQ(blockA, k, am, n, blockB, bm, ldm, hhVectors[k], (am + bm) - k);
	
}

/**
 * \brief Computes the \f$QR\f$ decomposition of the \f$b\times b\f$ portion of a matrix stored at matA
 *
 * \param matA A matrix in column-major order as a pointer to an array
 * \param m The number of rows in the matrix as stored
 * \param n The number of columns in the matrix as stored
 * \param lda The number of rows of the matrix as stored in memory
 * \returns A freshly allocated pointer to a matrix of pointers containing the householder reflectors from the decomposition.
 */
double** hhQR		(double* matA,
			int m,
			int n,
			int lda)//returns Q and triangularises matA in place.
{
	int k;
	double* xVect, **v = malloc(n * sizeof(double*));
	double* matQ = NULL;//, *matAgain;
	matQ = newMatrix(matQ, m, n);
	initMatrix(matQ, m, n, 2);

	for(k = 0; k < n; k++)
	{
		//x = matA(k:m,k)
		xVect = matA + CO(k,k,lda);//xVect is defo column vector from k -> b-k in column k of block

		//vk = sign(x[1])||x||_2e1 + x
		//vk = vk/||vk||_2
		v[k] = calcvk(xVect, m - k);

		//matA(k:ma,k:na) = matA(k:ma,k:na) - 2((vk*vk.T)/(vk.T*vk))*matA(k:ma,k:na)
		updateMatHHQRInPlace(matA, k, m, n, lda, v[k], m - k);
	}

	/*for(k = n - 1; k > -1; k --)
	{
		updateMatHHQRInPlace(matQ, k, m, n, b, n, v[k], m - k);
		free(v[k]);
	}*/
	printMatrix(matA, m, n, lda);
	//printMatrix(matQ, m, n);

	/*matAgain = multAB(matQ, m, n, m, matA, n, n, lda, 0);
	printMatrix(matAgain, m, n);*/
	
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

void updateDoubleQ	(double* matA,
			int koffset,
			int ma,
			int na,
			double* matB,
			int mb,
			int ldm,
			double* v,
			int l)
{
	int i, j, k, rows = l, cols = na-koffset, rowsA = l - mb;
	double y = 0, z, a;

	//Calculate y := -2/(sum(v[k]^2)) (lines 2-5 in Algorithm 1)
	for(k = 0; k < rows; k ++)
		y += (v[k] * v[k]);
	//y=1;
	y = -2/y;

	for(j = 0; j < cols; j ++)
	{
		//calculate z := sum(a(k,j) * v[k]) (lines 7 - 10 in Algorithm 1)
		z = 0;
		for(k = 0; k < rowsA; k ++)
			z += matA[CO((k+koffset),(j+koffset),ldm)] * v[k];
		for(; k < rows; k ++)
			z += matB[CO(((k-ma)+koffset),(j+koffset),ldm)] * v[k];

		//apply A(i,j) := A(i,j) + v[i] * y * z (lines 11 - 15 in Algorithm 1)
		for(i = 0; i < rowsA; i ++)
		{
			a = y * z;
			a *= v[i];
			matA[CO((i+koffset),(j+koffset),ldm)] += a;
		}
		for(; i < rows; i ++)
		{
			a = y * z;
			a *= v[i];
			matB[CO(((i-ma)+koffset),(j+koffset),ldm)] += a;
		}
	}
}/**
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

void updateSingleQ	(double* mat,
			int koffset,
			int m,
			int n,
			int ldm,
			double* v,
			int l)
{
	int i, j, k, rows = m-koffset, cols = n-koffset;
	double y = 0, z, a;

	//Calculate y := -2/(sum(v[k]^2)) (lines 2-5 in Algorithm 1)
	for(k = 0; k < rows; k ++)
		y += (v[k] * v[k]);
	//y=1;
	y = -2/y;

	for(j = 0; j < cols; j ++)
	{
		//calculate z := sum(a(k,j) * v[k]) (lines 7 - 10 in Algorithm 1)
		z = 0;
		for(k = 0; k < rows; k ++)
			z += mat[CO((k+koffset),(j+koffset),ldm)] * v[k];

		//apply A(i,j) := A(i,j) + v[i] * y * z (lines 11 - 15 in Algorithm 1)
		for(i = 0; i < rows; i ++)
		{
			a = y * z;
			a *= v[i];
			mat[CO((i+koffset),(j+koffset),ldm)] += a;
		}
	}
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

void updateMatHHQRInPlace(double* mat,
		int koffset,
		int m,
		int n,
		int ldm,
		double* v,
		int l)
{
	int i, j, k, rows = m-koffset, cols = n-koffset;
	double y = 0, z, a;

	//Calculate y := -2/(sum(v[k]^2)) (lines 2-5 in Algorithm 1)
	for(k = 0; k < rows; k ++)
		y += (v[k] * v[k]);
	//y=1;
	y = -2/y;

	for(j = 0; j < cols; j ++)
	{
		//calculate z := sum(a(k,j) * v[k]) (lines 7 - 10 in Algorithm 1)
		z = 0;
		for(k = 0; k < rows; k ++)
			z += mat[CO((k+koffset),(j+koffset),ldm)] * v[k];

		//apply A(i,j) := A(i,j) + v[i] * y * z (lines 11 - 15 in Algorithm 1)
		for(i = 0; i < rows; i ++)
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
void calcvkSingle	(double* x,
			int l,
			double* vk)
{
	int sign, i;
	double vk2norm, toadd, div;

	for(i = 0; i < l; i++)
		vk[i] = x[i];

	sign = vk[0] >= 0.0 ? 1 : -1;

	vk2norm = do2norm(vk, l);
	toadd = sign * vk2norm;
	div = 1/(x[0]+toadd);
	
	for(i = 1; i < l; i++)
		vk[i] *= div;

	vk[0] = 1;

}

/**
 * \brief Computes a Householder reflector from a pair of vectors from coupled blocks
 *
 * Calculates the Householder vector of the vector formed by a column in a pair of coupled blocks.
 * Assumes the vector is a square block on top of a rectangular (maybe square) block.
 * I.e that the length of the first vector is nonzero
 *
 * \param xa Pointer to the first array containing the first part of the column vector to compute the Householder reflector of
 * \param l The number of elements in \f$x\f$
 *
 * \returns A pointer to a newly allocated vector \f$v\f$, a Householder reflector of \f$x\f$
 */
void calcvkDouble	(double* xa,
			int ma,
			double* xb,
			int mb,
			int l,
			double* vk)
{
	int sign, i;
	double vk2norm, toadd, div;

	for(i = 0; i < ma; i++)
		vk[i] = xa[i];

	for(; i < l; i++)
		vk[i] = xb[i - ma];

	sign = vk[0] >= 0.0 ? 1 : -1;

	vk2norm = do2norm(vk, l);
	toadd = sign * vk2norm;
	div = 1/(xa[0]+toadd);
	
	for(i = 1; i < l; i++)
		vk[i] *= div;

	vk[0] = 1;
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
	double vk2norm, toadd, *vk, div;

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

double* multAB(double* matA, int ma, int na, int lda, double* matB, int mb, int nb, int ldb, int atrans)
{
	double* matC = NULL;
	int i, j, k;
	matC = newMatrix(matC, ma, nb);
	initMatrix(matC, ma, nb, 0);

	for(j = 0; j < nb; j++)
	{
		for(k = 0; k < na; k++)
		{
			for(i = 0; i < ma; i++)
			{
				matC[CO(i,j,ma)] += matA[CO(i,k,lda)] * matB[CO(k,j,ldb)];
			}
		}
	}

	return matC;
}

void printMatrix(double* mat, int m, int n, int ldm)
{
	int r, c;
	putchar('{');
	for(r = 0; r < m; r++)
	{
		putchar('{');
		for(c = 0; c < n; c++)
		{
			printf("%7.2f", mat[CO(r,c,ldm)]);
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
