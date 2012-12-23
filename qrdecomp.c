/**
 * @file qrdecomp.c
 * @author Sam Townsend
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "qrdecomp.h"

#define CO(i,j,m) ((m * j) + i)

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
	double* matA = NULL, *singleVector = NULL, *doubleVector = NULL;
	int ma = 4, na = 4, b = 2, i, j ,k, p = ma/b, q = na/b, minpq = p < q ? p : q;

	matA = newMatrix(matA, ma, na);

	srand(5);
	initMatrix(matA, ma, na, RAND);

	singleVector = newMatrix(singleVector, b, 1);

	doubleVector = newMatrix(doubleVector, 2*b, 1);

	printf("A:\n");
	printMatrix(matA, ma, na, ma);

	for(k = 0; k < minpq ; k ++)
	{
		qRSingleBlock(matA + CO((k*b),(k*b),ma), b, b, ma, singleVector);

		for(j = k + 1; j < q; j ++)
		{
			applySingleBlock(matA + CO((k*b),(j*b),ma), b, b, ma, matA + CO((k*b+1),(k*b),ma));
		}
		for(i = k + 1; i < p; i ++)
		{
			qRDoubleBlock(matA + CO((k*b),(k*b),ma), b, b, matA + CO((i*b),(k*b),ma), b, ma, doubleVector);
			printMatrix(matA, ma, na, ma);
			for(j = k + 1; j < q; j ++)
			{
				applyDoubleBlock(matA + CO((k*b),(j*b),ma), b, matA + CO((i*b),(j*b),ma), b, b,	ma, matA + CO((k*b),(k*b),ma), matA + CO((i*b),(k*b),ma));
			}
			printMatrix(matA, ma, na, ma);
		}
	}

	printf("tiled R:\n");
	printMatrix(matA, ma, na, ma);

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
			double* hhVector)
{
	int k;
	double* xVect;
	printf("factorising block:");
	printMatrix(block, m, n, ldb);

	for(k = 0; k < n; k++)
	{
		//x = matA(k:m,k)
		xVect = block + CO(k,k,ldb);//xVect is column vector from k -> b-k in column k of block
		//vk = sign(x[1])||x||_2e1 + x
		//vk = vk/||vk||_2
		calcvkSingle(xVect, m - k, hhVector);//returns essential
		/*printf("Householder vector %d: ", k);
		printMatrix(hhVector, m-k-1, 1, m-k-1);*/

		//matA(k:ma,k:na) = matA(k:ma,k:na) - 2((vk*vk.T)/(vk.T*vk))*matA(k:ma,k:na)
		updateSingleQ(block+CO(k,k,ldb), m - k, n - k, ldb, hhVector);

		//diag[k] = block[CO(k,k,ldb)];//update diag

		insSingleHHVector(block+CO(k+1,k,ldb), m - k - 1, hhVector);//replace column with essential part of vector
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
			double* hhVector)
{
	int k;
	double* xVectB, *xVectA;
	/*printf("blocks:\n");
	printMatrix(blockA, am, an, ldm);
	printMatrix(blockB, bm, an, ldm);*/

	for(k = 0; k < an; k++)
	{
		//x = matA(k:m,k)
		xVectA = blockA + CO(k,k,ldm);//xVect is column vector from k -> b-k in column k of block
		xVectB = blockB + CO(0,k,ldm);

		//vk = sign(x[1])||x||_2e1 + x
		//vk = vk/||vk||_2
		calcvkDouble(xVectA[0], am - k, xVectB, bm, (bm + am) - k, hhVector);//returns essential
		//printf("Householder vector %d: ", k);
		//printMatrix(hhVector, (am+bm)-k, 1, (am+bm)-k);
		hhVector = hhVector + 1;

		//matA(k:ma,k:na) = matA(k:ma,k:na) - 2(vk*(vk.T*matA(k:ma,k:na)
		updateDoubleQZeros(xVectA, am - k, an - k, xVectB, bm, ldm, hhVector, (am + bm) - k);//update top block

		//insSingleHHVector(xVectA, 1, hhVector);
		insSingleHHVector(xVectB, bm, hhVector + am - k - 1);
	}

	/*printMatrix(blockA, am, an, ldm);
	printMatrix(blockB, bm, an, ldm);*/
}

/**
 * \brief Applies precomputed householder vectors to a single block within a matrix
 * 
 * Applies the computed householder vectors to a single block of a matrix :
 * \f[Q_nQ_{n-1}\ldots Q_1A = Q^*A\f]
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
			double* hhVectors)
{
	int h, k;

	for(h = 0; h < n; h ++)
	{
		for(k = 0; k < n; k ++)
		{
			updateSingleQ(block + CO(k,h,ldb), m - k, 1, ldb, hhVectors + CO(k,k,ldb));
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
			double* hhVectorsA,
			double* hhVectorsB)
{
	int h, k;

	printf("applying:");
	printMatrix(hhVectorsA, am, n, ldm);
	printMatrix(hhVectorsB, bm, n, ldm);
	printf("to:");
	printMatrix(blockA, am, n, ldm);
	printMatrix(blockB, bm, n, ldm);

	for(h = 0; h < n; h ++)
	{//for each column of the result, x
		for(k = 0; k < n; k ++)
		{//multiply x by Q', overwriting with the result
			updateDoubleQ(blockA + CO(k,h,ldm), am - k, 1, blockB+CO(0,h,ldm), bm, ldm, hhVectorsB + CO(0,k,ldm), (am + bm) - k);
		}
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
			double* restv,
			int l)
{
	int i, j, k, cols = na;
	double z, a, y;
	printf("updating");
	printMatrix(matA, ma, na, ldm);
	printMatrix(matB, mb, na, ldm);
	printf("with");
	printMatrix(restv, mb, 1, ldm);

	y = 1;
	for(k = 0; k < mb; k ++)
	{
		y += restv[k] * restv[k];
	}

	y = (-2)/y;

	for(j = 0; j < cols; j ++)
	{
		//calculate z := sum(a(k,j) * v[k]) (lines 7 - 10 in Algorithm 1)
		//do for top portion(only non-zero element in top portion is firstv)
		z = matA[CO(0,j,ldm)];

		//then for lower portion
		for(k = 0; k < mb; k ++)
			z += matB[CO(k,j,ldm)] * restv[k];

		//apply A(i,j) := A(i,j) + v[i] * y * z for top portion (lines 11 - 15 in Algorithm 1)
		a = y * z;
		matA[CO(0,j,ldm)] += a;
		printf("%d %d: %5.2f\n ", 0,j, a);

		//then for the lower one
		for(i = 0; i < mb; i ++)
		{
			a = y * z;
			a *= restv[i];
			matB[CO(i,j,ldm)] += a;
		}
	}
	printMatrix(matB, mb, 1, ldm);
}

void updateDoubleQZeros	(double* matA,
			int ma,
			int na,
			double* matB,
			int mb,
			int ldm,
			double* v,
			int l)
{
	int i, j, k, rows = l, cols = na;
	double z, a, y;

	y = 1;
	for(k = 1; k < rows; k ++)
	{
		y += v[k-1] * v[k-1];
	}
	if(y != 0)
		y = (-2)/y;
	else
		y = 0;

	for(j = 0; j < cols; j ++)
	{
		//calculate z := sum(a(k,j) * v[k]) (lines 7 - 10 in Algorithm 1)
		z = matA[CO(0,j,ldm)];
		//do for top non-zero, non-diagonal portion
		for(k = 1; k < j+1; k ++)
		{
			z += matA[CO(k,j,ldm)] * v[k-1];
		}
		//count diagonal
		//z += diag[j] * v[k];
		//enter bottom portion (zeros for second half of top)
		k = l - mb;

		//then for lower
		for(; k < rows; k ++)
		{
			z += matB[CO((k-(l-mb)),j,ldm)] * v[k-1];
		}

		//apply A(i,j) := A(i,j) + v[i] * y * z for top portion (lines 11 - 15 in Algorithm 1)
		a = y * z;
		matA[CO(0,j,ldm)] += a;
		for(i = 1; i < j+1; i ++)
		{
			a = y * z;
			a *= v[i-1];
			matA[CO(i,j,ldm)] += a;
		}
		//matA[CO(i,j,ldm)] = diag[j] + y * z * v[i];
		
		i = l - mb;
		//then for the lower one
		for(; i < rows; i ++)
		{
			a = y * z;
			a *= v[i-1];
			matB[CO((i-ma),j,ldm)] += a;
		}
	}
}

/**
 * \brief Applies the householder update a single block
 * 
 * Computes the update: \f[a_{ij} = a_{ij} + \left(v_i\frac{-2}{\sum_{k=1}^mv_k^2}\right)\left(\sum_{k=1}^ma_{kj}v_k\right)\f] with the non-essential part
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
	double z, a, y;

	y = 1;
	for(k = 1; k < m; k ++)
	{
		y += v[k-1] * v[k-1];
	}
	y = (-2) / y;

	for(j = 0; j < n; j ++)
	{
		//calculate z := sum(a(k,j) * v[k]) (lines 7 - 10 in Algorithm 1)
		z = mat[CO(0,j,ldm)];
		for(k = 1; k < m; k ++)
			z += mat[CO(k,j,ldm)] * v[k-1];

		//apply A(i,j) := A(i,j) + v[i] * y * z (lines 11 - 15 in Algorithm 1)
		a = y * z;
		mat[CO(0,j,ldm)] += a;
		
		for(i = 1; i < m; i ++)
		{
			a = y * z;
			a *= v[i-1];
			mat[CO(i,j,ldm)] += a;
		}
	}
	//printMatrix(mat, m, n, ldm);
}

void insSingleHHVector	(double* block,
			int m,
			double* vector)
{
	int i;
	
	for(i = 0; i < m; i ++)
		block[i] = vector[i];
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
	double norm, div, beta;

	sign = x[0] >= 0.0 ? 1 : -1;
	beta = x[0];
	for(i = 1; i < l; i++)
		vk[i-1] = x[i];

	norm = do2norm(x, l);
	//printf("%5.2f ", norm);
	beta += norm * sign;

	//norm = do2norm(vk, l);
	//printf("%5.2f\n", vk[0]);

	if(norm != 0.0)
	{
		div = 1/beta;
	
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
void calcvkDouble	(double topDiag,
			int ma,
			double* xb,
			int mb,
			int l,
			double* vk)
{
	int sign, i;
	double norm, div;

	sign = topDiag >= 0.0 ? 1 : -1;
	vk[0] = topDiag;

	for(i = 1; i < ma; i++)
		vk[i] = 0;

	for(; i < l; i++)
		vk[i] = xb[i - ma];

	norm = do2norm(vk, l);
	//printf("%5.2f ", norm);
	vk[0] += norm * sign;
	//norm = do2norm(vk,l);
	//printf("%5.2f\n", vk[0]);

	if(norm != 0.0)
	{
		div = 1/vk[0];
	
		for(i = 1; i < l; i++)
			vk[i] *= div;
	}
	//vk[0] = 1;
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
		if(r != m-1)
			putchar('\n');
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
