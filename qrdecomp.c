/**
 * @file qrdecomp.c
 * @author Sam Townsend
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>

#include "include/gridscheduler.h"
#include "qrdecomp.h"

#define CO(i,j,m) ((m * j) + i)
#define COB(i,j,bs,N) ((i*N*bs) +(j*bs))

#define ZERO 0
#define RAND 1
#define EYE 2

#define NUMTHREADS 7

int main	(int argc,
		char* argv[])
{
	blockQR();

	return 0;
}

void blockQR()
{
	double* matA = NULL;
	int ma = 2048, na = 2048, b = 64, /*i, j ,k,*/ p = ma/b, q = na/b, t = 0, i, isDone/*, minpq = p < q ? p : q*/;
	pthread_t threads[NUMTHREADS];
	pthread_attr_t tattr;
	struct doTaskInfo taskInfs[NUMTHREADS];
	
	Task *taskGrid;
	Task curtasks[NUMTHREADS];
	
	double* workingVects[NUMTHREADS];
	
	matA = newMatrix(ma, na);

	srand(5);
	initMatrix(matA, ma, na, RAND);

	//printMatrix(matA, ma, na, ma);
	
	taskGrid = initScheduler(p, q);
	
	for(i = 0; i < NUMTHREADS; i ++)//fill items for every task
	{
		curtasks[i].taskStatus = NONE;
		threads[i] = 0;
		taskInfs[i].ptr = matA;
		taskInfs[i].b = b;
		taskInfs[i].ldm = ma;
		workingVects[i] = newMatrix(2*b,1);//single vector per thread, allocated here
		taskInfs[i].taskVect = workingVects[i];
	}

	printf("A:\n");

	pthread_attr_init(&tattr);
	pthread_attr_setdetachstate(&tattr, PTHREAD_CREATE_JOINABLE);

	isDone = getNextTask(&curtasks[t], taskGrid, p, q);

	while(isDone != 2)//some tasks are at least in the DOING state, not all done yet.
	{
		//t is a free task referencing task datas (curtasks, taskInfs, threads)
		if(isDone == 0)//new task was returned
		{
			taskInfs[t].t = curtasks[t];
			pthread_create(&threads[t], &tattr, pthr_doATask, (void*) &taskInfs[t]);
		}

		t = (t+1) % NUMTHREADS;

		while((threads[t] != 0)&&(pthread_kill(threads[t],0) != ESRCH))//if thread valid and not finished
		{
			t = (t+1) % NUMTHREADS;//find one that is finished
		}

		if(threads[t] != 0)
			pthread_detach(threads[t]);//free resources

		//iterate through threads until t references empty process
		//if associated task is done
			//finish task

		if(curtasks[t].taskStatus != NONE)//not one of the beginning tasks
		{
			doneATask(taskGrid, p, q, curtasks[t]);//register finished
		}

		isDone = getNextTask(&curtasks[t], taskGrid, p, q);
	}
	
	/*for(k = 0; k < minpq ; k ++)
	{
		//compute QR of Akk: Akk <-- Vkk,Rkk
		qRSingleBlock(matA + CO((k*b),(k*b),ma), b, b, ma);

		for(j = k + 1; j < q; j ++)
		{
			//along kth row
			//apply Vkk: Akj <-- Qkk'*Akj
			applySingleBlock(matA + CO((k*b),(j*b),ma),
					b, b, ma,
					matA + CO((k*b),(k*b),ma));//vectors start below diagonal
		}
		for(i = k + 1; i < p; i ++)
		{
			//down kth column
			//compute QR of Akk coupled with Aik: Akk, Aik <-- R~kk,Vik
			qRDoubleBlock	(matA + CO((k*b),(k*b),ma),
					b, b,
					matA + CO((i*b),(k*b),ma),
					b, ma);

			for(j = k + 1; j < q; j ++)
			{
				//along ith and kth rows
				//apply Vik to coupled blocks: Akj, Aij <-- Qik'*(Akj,Aij)
				applyDoubleBlock(matA + CO((k*b),(j*b),ma),
						b,
						matA + CO((i*b),(j*b),ma),
						b, b, ma,
						matA + CO((i*b),(k*b),ma));
			}
		}
	}*/

	printf("tiled R:\n");
	//printMatrix(matA, ma, na, ma);

	deleteMatrix(matA);
	free(taskGrid);
	pthread_attr_destroy(&tattr);
}

void* pthr_doATask(void* taskInfoptr)
{
	struct doTaskInfo localTaskInf = *((struct doTaskInfo*)taskInfoptr);
	doATask(localTaskInf.t, localTaskInf.ptr, localTaskInf.b, localTaskInf.ldm, localTaskInf.taskVect);
	pthread_exit(NULL);
}

void doATask(Task t, double* mat, int b, int ldm, double* colVect)
{
	double *blockV, *blockA, *blockB;

	switch(t.taskType)
	{
		case QRS:
		{
			blockV = mat + CO((t.k*b),(t.k*b),ldm);
			qRSingleBlock(blockV, b, b, ldm, colVect);
			//printf("qr %d,%d\n", t.k, t.k);
			break;
		}
		case SAPP:
		{
			blockV = mat + CO((t.k*b),(t.k*b),ldm);
			blockA = mat + CO((t.k*b),(t.m*b),ldm);
			applySingleBlock(blockA, b, b, ldm, blockV);
			//printf("sapp %d,%d %d,%d\n", t.k, t.k, t.k, t.m);
			break;
		}
		case QRD:
		{
			blockA = mat + CO((t.k*b),(t.k*b),ldm);
			blockB = mat + CO((t.l*b),(t.k*b),ldm);
			qRDoubleBlock(blockA, b, b, blockB, b, ldm, colVect);
			//printf("qrd %d,%d %d,%d\n", t.k, t.k, t.l, t.k);
			break;
		}
		case DAPP:
		{
			blockV = mat + CO((t.l*b),(t.k*b),ldm);
			blockA = mat + CO((t.k*b),(t.m*b),ldm);
			blockB = mat + CO((t.l*b),(t.m*b),ldm);
			applyDoubleBlock(blockA, b, blockB, b, b, ldm, blockV);
			//printf("dapp %d,%d %d,%d %d,%d\n", t.l, t.k, t.k, t.m, t.l, t.m);
			break;
		}
	}
}

/**
 * \brief Computes the QR decomposition of a single block within a matrix
 *
 * \param block A pointer to the first element of the block
 * \param m The number of rows in the block
 * \param n The number of columns in the block
 * \param ldb The leading dimension of the matrix
 * \param hhVectors A pointer to a pre-allocated array for use as a scratchpad to store the householder vector
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
//	double* hhVector = newMatrix(m-1, 1);

	/*printf("input to QRS:\n");
	printMatrix(block, m, n, ldb);*/
	for(k = 0; k < n; k++)
	{
		//x = matA(k:m,k)
		xVect = block + CO(k,k,ldb);//xVect is column vector from k -> b-k in column k of block

		//vk = sign(x[1])||x||_2e1 + x
		//vk = vk/vk(0)
		calcvkSingle(xVect, m - k, hhVector);//returns essential

		//matA(k:ma,k:na) = matA(k:ma,k:na) - 2((vk*vk.T)/(vk.T*vk))*matA(k:ma,k:na)
		updateSingleQ(xVect, m - k, n - k, ldb, hhVector);

		//replace column with essential part of vector	
		insSingleHHVector(block+CO(k+1,k,ldb), m - k - 1, hhVector);
	}
	//deleteMatrix(hhVector);
}

/**
 * \brief Computes the QR decomposition of the matrix formed by coupling two blocks
 * 
 * Computes the QR decomposition of a rectangular block formed by coupling two blocks
 * from within the same matrix on top of each other.
 * Stores the R factor in place and stores the essential parts of the householder
 * vectors in a block overwriting the bottom block.
 *
 * \param blockA A pointer to the first element of the "top" block
 * \param am The number of rows in blockA
 * \param an The number of columns in the coupled matrix
 * \param blockB A pointer to the first element of the "bottom" block
 * \param bm The number of rows in blockB
 * \param ldm The leading dimension of the main matrix
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
	//double* hhVector = newMatrix(am + bm, 1), *freeThisptr = hhVector;
	/*printf("input to QRD:\n");
	printMatrix(blockA, am, an, ldm);
	printMatrix(blockB, bm, an, ldm);*/

	for(k = 0; k < an; k++)
	{
		//x = matA(k:m,k)
		xVectA = blockA + CO(k,k,ldm);//xVect is column vector from k -> b-k in column k of block
		xVectB = blockB + CO(0,k,ldm);

		//vk = sign(x[1])||x||_2e1 + x
		//vk = vk/vk[0]
		calcvkDouble(xVectA[0], am - k, xVectB, (bm + am) - k, hhVector);//returns essential
		hhVector = hhVector + 1;

		//matA(k:ma,k:na) = matA(k:ma,k:na) - (2/(vk.T*vk))*vk*(vk.T*matA(k:ma,k:na)
		//update both blocks, preserving the vectors already stored below the diagonal in the top block and treating them as if they were zeros.
		updateDoubleQZeros(xVectA, am - k, an - k, xVectB, bm, ldm, hhVector, (am + bm) - k);

		//place the kth vector in place overwriting the bottom block
		insSingleHHVector(xVectB, bm, hhVector + am - k - 1);
	}
	//deleteMatrix(freeThisptr);
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
 * 
 * \returns void
 */
void applySingleBlock	(double* block,
			int m,
			int n,
			int ldb,
			double* hhVectors)
{
	int h, k;
	/*printf("input to SAPP:\n");
	printMatrix(hhVectors, m, n, ldb);
	printMatrix(block, m, n, ldb);*/

	for(h = 0; h < n; h ++)
	{
		//construct the result column by column
		for(k = 0; k < n; k ++)
		{
			//apply each successive kth vector to each hth column of the block to form the result
			updateSingleQInp(block + CO(k,h,ldb),
					m - k, 1, ldb,
					hhVectors + CO(k,k,ldb));
		}
	}
	//printMatrix(block, m, n, ldb);
}

/**
 * \brief Applies precomputed householder vectors to a rectangular block formed from two blocks within the same matrix
 * 
 * Applies the n precomputed householder vectors to the rectangular matrix
 * formed by coupling blockA on top of blockB.
 * Computes \f$Q^*\begin{bmatrix}B_a\\B_b\end{bmatrix}\f$
 *
 * \param blockA A pointer to the first element of the "top" block in the coupling
 * \param am The number of rows in blockA
 * \param blockB A pointer to the first element of the "bottom" block in the coupling
 * \param bm The number of rows in the "bottom" block
 * \param n The number of columns in the coupling
 * \param ldm The leading dimension of the matrix where both blocks reside
 * \param hhVectorsB A pointer to the essential portions of a block of vectors to apply
 *
 * \returns void
 */
void applyDoubleBlock	(double* blockA,
			int am,
			double* blockB,
			int bm,
			int n,
			int ldm,
			double* hhVectorsB)
{
	int h, k;
	/*printf("input to DAPP:\n");
	printMatrix(hhVectorsB, am, n, ldm);
	printMatrix(blockA, am, n, ldm);
	printMatrix(blockB, am, n, ldm);*/

	for(h = 0; h < n; h ++)
	{//for each column of the result, x
		for(k = 0; k < n; k ++)
		{//multiply x by Q', overwriting with the result
			updateDoubleQ(blockA + CO(k,h,ldm), am - k, 1, blockB+CO(0,h,ldm), bm, ldm, hhVectorsB + CO(0,k,ldm));
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
 * 
 * \returns void
 */

void updateDoubleQ	(double* matA,
			int ma,
			int na,
			double* matB,
			int mb,
			int ldm,
			double* v)
{
	int i, j, k, cols = na;
	double z, a, y;

	//compute y as per Algorithm 1, assuming the first element of v is 1 from the earlier normalisation
	y = 1;
	for(k = 0; k < mb; k ++)
	{
		y += v[k] * v[k];
	}
	y = (-2)/y;

	for(j = 0; j < cols; j ++)
	{
		//calculate z := sum(a(k,j) * v[k]) (lines 7 - 10 in Algorithm 1)
		//start with the single relevant top element and v[0]=1
		z = matA[CO(0,j,ldm)];

		//then for lower portion, using the vector contents
		for(k = 0; k < mb; k ++)
			z += matB[CO(k,j,ldm)] * v[k];

		//apply A(i,j) := A(i,j) + v[i] * y * z for top portion (lines 11 - 15 in Algorithm 1)
		a = y * z;
		matA[CO(0,j,ldm)] += a;

		//then for the lower one
		for(i = 0; i < mb; i ++)
		{
			a = y * z;
			a *= v[i];
			matB[CO(i,j,ldm)] += a;
		}
	}
}

/**
 * \brief Applies the householder update to a coupled matrix, not touching the top portion below the diagonal
 * 
 * Computes the update: \f[a_{ij} = a_{ij} + \left(v_i\frac{-2}{\sum_{k=1}^mv_k^2}\right)
 * \left(\sum_{k=1}^ma_{kj}v_k\right)\f] for a matrix constructed by coupling two blocks
 * where the top block is not changed below the diagonal, preserving vectors already stored there.
 *
 * \param matA Top portion of coupling
 * \param ma The number of rows in the top block
 * \param na The number of columns in the coupling 
 * \param matB Lower portion of coupling
 * \param mb The number of rows in the lower portion
 * \param ldm The leading dimension of the matrix in memory
 * \param v Pointer to the householder vector
 * \param l The length of this vector
 * 
 * \returns void
 */


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
 * Computes the update: \f[a_{ij} = a_{ij} + \left(v_i\frac{-2}{\sum_{k=1}^mv_k^2}\right)\left(\sum_{k=1}^ma_{kj}v_k\right)\f] with the non-essential part. see algorithm 1 for details
 * 
 * \param mat Matrix of size \f$m \times n\f$
 * \param m The number of rows in the matrix
 * \param n The number of columns in the matrix
 * \param ldm The leading dimension of mat in memory
 * \param v Pointer to an array containing the Householder reflector \f$v\f$
 *
 * \returns void
 */
void updateSingleQInp	(double* mat,
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
		y += v[k] * v[k];
	}
	y = (-2) / y;

	for(j = 0; j < n; j ++)
	{
		//calculate z := sum(a(k,j) * v[k]) (lines 7 - 10 in Algorithm 1)
		z = mat[CO(0,j,ldm)];
		for(k = 1; k < m; k ++)
			z += mat[CO(k,j,ldm)] * v[k];

		//apply A(i,j) := A(i,j) + v[i] * y * z (lines 11 - 15 in Algorithm 1)
		a = y * z;
		mat[CO(0,j,ldm)] += a;
		
		for(i = 1; i < m; i ++)
		{
			a = y * z;
			a *= v[i];
			mat[CO(i,j,ldm)] += a;
		}
	}
}
/**
 * \brief Applies the householder update a single block
 * 
 * Computes the update: \f[a_{ij} = a_{ij} + \left(v_i\frac{-2}{\sum_{k=1}^mv_k^2}\right)\left(\sum_{k=1}^ma_{kj}v_k\right)\f] with the non-essential part. see algorithm 1 for details
 * 
 * \param mat Matrix of size \f$m \times n\f$
 * \param m The number of rows in the matrix
 * \param n The number of columns in the matrix
 * \param ldm The leading dimension of mat in memory
 * \param v Pointer to an array containing the Householder reflector \f$v\f$
 *
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
}
/**
 * \brief Copies the vector of length m into block
 *
 * \param block The pointer to copy the vector into
 * \param m The length of the vector
 * \param vector The vector to be copied
 * \returns void
 */
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
 * Then does a non-standard normalisation \f$v\f$: \f[v = \frac{v}{v_1}\f]
 * 
 * \param x Pointer to an array containing a column vector to compute the Householder reflector of
 * \param l The number of elements in \f$x\f$
 * \param vk A pointer to an allocated array to store the resulting vector of size l - 1
 * due to the implied 1 as the first element
 *
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
	//copy the values
	for(i = 1; i < l; i++)
		vk[i-1] = x[i];

	//take the euclidian norm of the original vector
	norm = do2norm(x, l);
	//calculate the new normalisation
	beta += norm * sign;

	if(norm != 0.0)
	{
		//normalise 
		div = 1/beta;
	
		for(i = 0; i < l-1; i++)
			vk[i] *= div;
	}
}

/**
 * \brief Computes a Householder reflector from a pair of vectors from coupled blocks
 *
 * Calculates the Householder vector of the vector formed by a column in a pair of coupled blocks.
 * There is a single non-zero element, in the first row, of the top vector. This is passed as topDiag
 *
 * \param topDiag The only non-zero element of the incoming vector in the top block
 * \param ma The number of elements in the top vector
 * \param xb Pointer to the lower vector
 * \param l The number of elements in the whole vector
 * \param vk A pointer to a pre-allocated array to store the householder vector of size l
 *
 * \returns void
 */
void calcvkDouble	(double topDiag,
			int ma,
			double* xb,
			int l,
			double* vk)
{
	int sign, i;
	double norm, div;
	//same non-standard normalisation as for single blocks above, but organised without a temporary beta veriable

	sign = topDiag >= 0.0 ? 1 : -1;
	vk[0] = topDiag;
	//use vk[0] as beta
	for(i = 1; i < ma; i++)
		vk[i] = 0;

	for(; i < l; i++)
		vk[i] = xb[i - ma];

	norm = do2norm(vk, l);
	vk[0] += norm * sign;

	if(norm != 0.0)
	{
		div = 1/vk[0];
	
		for(i = 1; i < l; i++)
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

double* multAB(double* matA, int ma, int na, int lda, double* matB, int nb, int ldb)
{
	double* matC = NULL;
	int i, j, k;
	matC = newMatrix(ma, nb);
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

double* newMatrix(int m, int n)
{
	double* matptr;
	matptr = malloc(m * n * sizeof(double));
	return matptr;
}
