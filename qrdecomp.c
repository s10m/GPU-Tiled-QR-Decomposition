/**
 * @file qrdecomp.c
 * @author Sam Townsend
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "qrdecomp.h"

#define CO(i,j,m) ((m * j) + i)
#define COB(i,j,m) (sizeof(double)*(((m)*(j) + (i))))
#define ZERO 0
#define RAND 1
#define EYE 2

int main	(int argc,
		char* argv[])
{
	blockQR();

	return 0;
}

void blockQR()
{
	double* matA = NULL,* matAs = NULL, *matQs = NULL, *matAd = NULL, *matQd = NULL, **singleVectors, **doubleVectors;
	int ma = 8, na = 4, b = 2, i, j ,k, p = ma/b, q = na/b, minpq = p < q ? p : q;

	matAs = newMatrix(matAs, ma, na);
	matAd = newMatrix(matAd, ma, na);
	matQs = newMatrix(matQs, ma, ma);
	matQd = newMatrix(matQd, ma, ma);
	matA = newMatrix(matA, ma, na);

	srand(5);
	initMatrix(matA, ma, na, RAND);
	srand(5);
	initMatrix(matAs, ma, na, RAND);
	initMatrix(matQs, ma, ma, EYE);
	srand(5);
	initMatrix(matAd, ma, na, RAND);
	initMatrix(matQd, ma, ma, EYE);

	printMatrix(matAs, ma, na, ma);
	printMatrix(matAd, ma, na, ma);
	allocVectors(&singleVectors, ma, na);
	allocVectors(&doubleVectors, ma, na);//2*b, b);
	printf("A:");
	printMatrix(matA, ma, na, ma);

	for(k = 0; k < minpq ; k ++)
	{
		qRSingleBlock(matA + CO((k*b),(k*b),ma), b, b, ma, singleVectors);

		for(j = k + 1; j < q; j ++)
		{
			applySingleBlock(matA + CO((k*b),(j*b),ma), b, b, ma, singleVectors);
		}
		printMatrix(matA, ma, na, ma);
		
		for(i = k + 1; i < p; i ++)
		{
			qRDoubleBlock(matA + CO((k*b),(k*b),ma), b, b, matA + CO((i*b),(k*b),ma), b, ma, doubleVectors);
			
			for(j = k + 1; j < q; j ++)
			{
				applyDoubleBlock(matA + CO((k*b),(j*b),ma), b, matA + CO((i*b),(j*b),ma), b, b, ma, doubleVectors);
			}
		}
	}
	/*qRDoubleBlock(matAd, 4, 4, matAd + 4, 4, ma, doubleVectors);
	applySingleBlock(matQd, ma, na, ma, doubleVectors);*/

	qRSingleBlock(matAs, ma, na, ma, singleVectors);
	applySingleBlock(matQs, ma, na, ma, singleVectors);
	
	/*printf("double: ");
	printMatrix(matAd, ma, na, ma);
	printMatrix(matQd, ma, ma, ma);
	printMatrix(multAB(matQd, ma, ma, ma, matAd, ma, na, ma, 0), ma, na, ma);*/

	printf("single: ");
	printMatrix(matAs, ma, na, ma);
	printMatrix(matQs, ma, ma, ma);
	printMatrix(multAB(matQs, ma, ma, ma, matAs, ma, na, ma, 0), ma, na, ma);
	
	printf("tiled: ");
	printMatrix(matA, ma, na, ma);

	deleteMatrix(matAs);
	deleteMatrix(matAd);
	deleteMatrix(matA);
}

/**
 * \brief Allocates storage for n householder vectors in a triangular array
 * \param hhVectors A pointer to the location to allocate the storage
 * \param m The number of rows to allocate for
 * \param n The number of vectors to allocate
 * \returns void
 */
void allocVectors	(double*** hhVectors,
			int m,
			int n)
{
	int i;
	
	*hhVectors = malloc(n * sizeof(double*));
	
	for(i = 0; i < n; i++)
	{
		(*hhVectors)[i] = malloc((m - i) * sizeof(double));
	}
}

/**
 * \brief Computes the QR decomposition of a single block within a matrix
 *
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

	//printMatrix(block, m, n, ldb);

	for(k = 0; k < n; k++)
	{
		//x = matA(k:m,k)
		xVect = block + CO(k,k,ldb);//xVect is column vector from k -> b-k in column k of block
		//vk = sign(x[1])||x||_2e1 + x
		//vk = vk/||vk||_2
		calcvkSingle(xVect, m - k, hhVectors[k]);
		/*printf("Householder vector %d: ", k);
		printMatrix(hhVectors[k], m-k, 1, m-k);*/

		//matA(k:ma,k:na) = matA(k:ma,k:na) - 2((vk*vk.T)/(vk.T*vk))*matA(k:ma,k:na)
		updateSingleQ(block+CO(k,k,ldb), m-k, n-k, ldb, hhVectors[k]);
		//updateDoubleQ(block+CO(k,k,ldb), -k, 
	}

	//printMatrix(block, m, n, ldb);
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
 * \param bm The numbeallocVectors(&singleVectors, b, b);r of rows in blockB
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

	/*printMatrix(blockA, am, an, ldm);
	printMatrix(blockB, bm, an, ldm);*/

	for(k = 0; k < an; k++)
	{
		//x = matA(k:m,k)
		xVectA = blockA + CO(k,k,ldm);//xVect is column vector from k -> b-k in column k of block
		xVectB = blockB + CO(0,k,ldm);

		//vk = sign(x[1])||x||_2e1 + x
		//vk = vk/||vk||_2
		calcvkDouble(xVectA, am - k, xVectB, bm, (bm + am) - k, hhVectors[k]);
		/*printf("Householder vector %d: ", k);
		printMatrix(hhVectors[k], (am+bm)-k, 1, (am+bm)-k);*/

		//matA(k:ma,k:na) = matA(k:ma,k:na) - 2((vk*vk.T)/(vk.T*vk))*matA(k:ma,k:na)
		updateDoubleQ(xVectA, am - k, an - k, xVectB, bm, ldm, hhVectors[k], (am + bm) - k);//update top block
	}

	/*printMatrix(blockA, am, an, ldm);
	printMatrix(blockB, bm, an, ldm);*/
}

/** 
 * \brief Applies precomputed householder vectors to a single block within a matrix
 * 
 * Applies the computed householder vectors to a single block of a matrix :
 * \f[Q_1Q_2\ldots Q_nM = QM\f]
 *
 * \param block A pointer to the first element of the block to apply the vectors to
 * \param m The number of rows in the block
 * \param n The number of columns in the block
 * \param ldb The leading dimension of the main matrix
 * \param hhVectors A pointer to a triangular array containing n householder vectors
 * \returns void
 */
void applySingleBlock	(double* block,
			int m,
			int n,
			int ldb,
			double** hhVectors)
{
	int h, k;
	for(h = 0; h < m; h ++)
	{
		for(k = 0; k < n; k ++)
		{
			updateSingleQ(block + CO(k,h,ldb), m - k, 1, ldb, hhVectors[k]);
		}
	}
}

/**
 * \brief Applies precomputed householder vectors to a rectangular block formed from two blocks within the same matrix
 * 
 * Applies the n precomputed householder vectors to the rectangular matrix
 * formed by coupling blockA on top of blockB.
 * Computes \f$Q\begin{bmatrix}B_a\\B_b\end{bmatrix}\f$
 *
 * \param blockA A pointer to the first element of the "top" block in the coupling
 * \param am The number of rows in blockA
 * \param blockB A pointer to the first element of the "bottom" block in the coupling
 * \param bm The number of rows in the "bottom" block
 * \param n The number of columns in the coupling
 * \param ldm The leading dimension of the matrix where both blocks reside
 * \param hhVectors A pointer to a triangular array containing n householder vectors
 * \returns void
 */
void applyDoubleBlock	(double* blockA,
			int am,
			double* blockB,
			int bm,
			int n,
			int ldm,
			double** hhVectors)
{
	int h, k;
	for(h = 0; h < am; h ++)
	{
		for(k = 0; k < n; k ++)
			updateDoubleQ(blockA + CO(k,h,ldm), am - k, 1, blockB+CO(0,h,ldm), bm, ldm, hhVectors[k], (am + bm) - k);
	}
	
}

/**
 * \brief Applies the householder update to a coupled matrix
 * 
 * Computes the update: \f[a_{ij} = a_{ij} + \left(v_i\frac{-2}{\sum_{k=1}^mv_k^2}\right)\left(\sum_{k=1}^ma_{kj}v_k\right)\f] for a matrix constructed by coupling two blocks
 *
 * \param matA Top portion of coupling
 * \param ma The number of rows in the top block
 * \param na The number of columns in the coupling 
 * \param matB Lower portion of coupling
 * \param mb The number of rows in the lower portion
 * \param ldm The leading dimension of the matrix in memory
 * \param v Pointer to an array containing the Householder reflector
 * \param l The number of elements in v
 * \returns void
 */

void updateDoubleQ	(double* matA,
			int ma,
			int na,
			double* matB,
			int mb,
			int ldm,
			double* v,
			int l)
{
	int i, j, k, rows = ma + mb, cols = na, rowsA = ma;
	double z, a;

	for(j = 0; j < cols; j ++)
	{
		//calculate z := sum(a(k,j) * v[k]) (lines 7 - 10 in Algorithm 1)
		z = 0;
		//do for top portion
		for(k = 0; k < rowsA; k ++)
			z += matA[CO(k,j,ldm)] * v[k];
		//then for lower
		for(; k < rows; k ++)
			z += matB[CO((k-ma),j,ldm)] * v[k];

		//apply A(i,j) := A(i,j) + v[i] * y * z for top portion (lines 11 - 15 in Algorithm 1)
		for(i = 0; i < rowsA; i ++)
		{
			a = -2 * z;
			a *= v[i];
			matA[CO(i,j,ldm)] += a;
		}
		//then for the lower one
		for(; i < rows; i ++)
		{
			a = -2 * z;
			a *= v[i];
			matB[CO((i-ma),j,ldm)] += a;
		}
	}
}

/**
 * \brief Applies the householder update a single block
 * 
 * Computes the update: \f[a_{ij} = a_{ij} + \left(v_i\frac{-2}{\sum_{k=1}^mv_k^2}\right)\left(\sum_{k=1}^ma_{kj}v_k\right)\f]
 * 
 * \param mat Matrix of size \f$m \times n\f$
 * \param m The number of rows in the matrix
 * \param n The number of columns in the matrix
 * \param ldm The leading dimension of mat in memory
 * \param v Pointer to an array containing the Householder reflector \f$v\f$
 * \returns void
 */
void updateSingleQ	(double* mat,
			int m,
			int n,
			int ldm,
			double* v)
{
	int i, j, k;
	double z, a;

	for(j = 0; j < n; j ++)
	{
		//calculate z := sum(a(k,j) * v[k]) (lines 7 - 10 in Algorithm 1)
		z = 0;
		for(k = 0; k < m; k ++)
			z += mat[CO(k,j,ldm)] * v[k];

		//apply A(i,j) := A(i,j) + v[i] * y * z (lines 11 - 15 in Algorithm 1)
		for(i = 0; i < m; i ++)
		{
			a = -2 * z;
			a *= v[i];
			mat[CO(i,j,ldm)] += a;
		}
	}
	//printMatrix(mat, m, n, ldm);
}

/**
 * \brief Computes a Householder reflector \f$v\f$ of a vector \f$x\f$ for a single block
 *
 * Computes: \f[v = \textrm{sign}(x_1)||x||_2e_1+x\f]
 * Then normalises \f$v\f$: \f[v = \frac{v}{||v||_2}\f]
 * 
 * \param x Pointer to an array containing a column vector to compute the Householder reflector of
 * \param l The number of elements in \f$x\f$
 * \param vk A pointer to an allocated array to store the resulting vector
 * \returns void
 */
void calcvkSingle	(double* x,
			int l,
			double* vk)
{
	int sign, i;
	double norm, div;

	sign = x[0] >= 0.0 ? 1 : -1;

	for(i = 0; i < l; i++)
		vk[i] = x[i];

	norm = do2norm(vk, l);
	printf("%5.2f ", norm);
	vk[0] += norm * sign;

	norm = do2norm(vk, l);
	printf("%5.2f\n", vk[0]);

	if(norm != 0.0)
	{
		div = 1/norm;
	
		for(i = 0; i < l; i++)
			vk[i] *= div;
	}
}

/**
 * \brief Computes a Householder reflector from a pair of vectors from coupled blocks
 *
 * Calculates the Householder vector of the vector formed by a column in a pair of coupled blocks.
 * Assumes the vector is a square block on top of a rectangular (maybe square) block.
 * I.e that the length of the first vector is nonzero
 *
 * \param xa Pointer to the first array containing the first part of the column vector to compute the Householder reflector of
 * \param ma The number of elements in the top vector
 * \param xb Pointer to the lower vector
 * \param mb The number of elements in the lower vector
 * \param l The number of elements in the householder vector
 * \param vk A pointer to a pre-allocated array to store the householder vector in
 *
 * \returns void
 */
void calcvkDouble	(double* xa,
			int ma,
			double* xb,
			int mb,
			int l,
			double* vk)
{
	int sign, i;
	double norm, div;

	sign = xa[0] >= 0.0 ? 1 : -1;

	for(i = 0; i < ma; i++)
		vk[i] = xa[i];

	for(; i < l; i++)
		vk[i] = xb[i - ma];

	norm = do2norm(vk, l);
	printf("%5.2f ", norm);
	vk[0] += norm * sign;
	norm = do2norm(vk,l);
	printf("%5.2f\n", vk[0]);

	if(norm != 0.0)
	{
		div = 1/norm;
	
		for(i = 0; i < l; i++)
			vk[i] *= div;
	}
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
	putchar('[');
	for(r = 0; r < m; r++)
	{
		for(c = 0; c < n; c++)
		{
			printf(" %7.2f", mat[CO(r,c,ldm)]);
		}
		putchar(';');
	}
	printf("]\n");
}

void initMatrix(double* mat, int m, int n, int mode)
{
	int r, c;

	for(c = 0; c < n; c++)
	{
		for(r = 0; r < m; r++)
		{
			if(mode == ZERO)
				mat[CO(r,c,m)] = 0;
			else if(mode == RAND)
				mat[CO(r,c,m)] = rand() % (m * n);
			else if(mode == EYE)
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
