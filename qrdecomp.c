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
//#include "include/gpucalc.h"
#include "include/cycle.h"
#include "qrdecomp.h"

#define CO(i,j,m) ((m * j) + i)
#define COB(i,j,bs,N) ((i*N*bs) +(j*bs))

#define NUMTHREADS 8

#define EPSILON 0.001

enum {ZERO, RAND, RANDZO, EYE};

int main	(int argc,
		char* argv[])
{
	//Declare 64x64 matrix, random data in all.
	
	/*float results[10000];
	double avg;
	const int tests = 100;
	int r, b;

	for(b = 1; b < 129; b ++)
	{
		testDAPP(results, tests, b);
		avg = 0.0;
		for(r = 0; r < tests; r ++)
		{
			//printf("%5f\n", results[r]/b);
			avg += results[r]/b;
		}
		printf("%d: %2f\n", b, avg/tests);
	}*/
	int numtests, mtiles, ntiles;

	//numtests = argc > 1 ? atoi(argv[1]) : 1;
	mtiles = argc > 1 ? atoi(argv[1]) : 4;
	ntiles = argc > 1 ? atoi(argv[1]) : 4;

	tiledQR( numtests, mtiles, ntiles );

	return 0;
}

/* Generates a random MATROWTILES*MATCOLTILES 32x32 tile matrix, performs the
   decomposition on the CPU once, then performs numtests runs on the GPU with
   the same data. 
   Counts the number of times the outputs are the same and differ, also gathers
   basic timing information. */
void tiledQR( int numtests, int mtiles, int ntiles )
{
	float* 	matCPU = NULL, *matComp = NULL, *matData = NULL, *matTau = NULL;

	int 	b = 32, m = mtiles*b, n = ntiles*b, ldm = m,
		times = 0, failures = 0, successes = 0,
		currentM;

	float 	totaltime = 0;

	ticks 	tick, tock;

	matCPU 	= newMatrix(m, n);
	matData = newMatrix(m, n);
	matTau 	= newMatrix(m, n);
	matComp = newMatrix(m, n);

	srand(5);

	//for(currentM = mtiles; currentM <= mtiles; currentM ++)
	//{
		//m = currentM * b;
		//n = currentM * b;
		//printf("\n(%d, %d):\n", m, n);

		initMatrix(matData, m, n, ldm, RANDZO);
		initMatrix(matTau, m, n, ldm, ZERO);

		copyMatrix(matData, m, n, ldm, matCPU);
	
		taskQRP_threads	(matData, matCPU, matTau,
				m, n, b, ldm,
				1);
		//printf("done CPU\n");
	
		/*while( times ++ < numtests )
		{*/
			copyMatrix(matData, m, n, ldm, matComp);
			//printf("%d: ", times);
			cudaQRTask(matComp, m, n, ldm, 128);

			if( ! checkEqual(matComp, matCPU, m, n, ldm) )
			{
				printf("Failure.\n");
				failures ++;
			}
			else
			{
				printf("Correct.\n");
				successes ++;
			}
		//}
		times = 0;

		//printf("%d failures and %d successes out of %d.\n", failures, successes, numtests);
		
		failures = 0;
		successes = 0;

		cudaDeviceReset();
	//}

	deleteMatrix(matCPU);
	deleteMatrix(matComp);
	deleteMatrix(matData);
	deleteMatrix(matTau);
}

/**
 * Performs a tiled task based QR decomposition on the mxn matrix matData,
 * returning the result into matResult.
 *
 * \param matData The mxn matrix containing the input values for the decomposition.
 * \param matResult The mxn matrix to return the result into.
 * \param tau The mxn matrix representing the tau matrix returned by the transformation.
 * \param m The number of rows in the matrices.
 * \param n The number of columns in the matrices.
 * \param b The block size in the matrices. Assumed to be some factor of m and n.
 * \param useWY A boolean int representing whether the function should use the
 *              WY representation to perform the computation or not.
 */
void taskQRP_threads	(float* matData, float* matResult,
			float* tau,
			int m, int n, int b, int ldm,
			int useWY)
{
	ticks 			tick, tock;

	pthread_t 		threads[NUMTHREADS];
	pthread_attr_t 		tattr;

	pthread_cond_t 		tCond = PTHREAD_COND_INITIALIZER;
	pthread_mutex_t 	tMutex = PTHREAD_MUTEX_INITIALIZER;

	struct 	ThreadInfo 	threadInfs[NUMTHREADS];
	
	Task 			*taskGrid;

	int 			p = m / b, q = n / b,
				tCondMet = 0,
				i, t;
	
	/* Copy the data into the destination, to perform the decomposition in place. */
	copyMatrix(matData, m, n, ldm, matResult);

	/* Initialise the task scheduler. */
	taskGrid = initScheduler(p, q);

	/* Fill the structs for the threads with all the per-thread resources. */
	for(i = 0; i < NUMTHREADS; i ++)
	{
		threadInfs[i].wspace[0] = newMatrix(2*b,1);
		threadInfs[i].wspace[1] = newMatrix(2*b,1);
		threadInfs[i].mat = matResult;
		threadInfs[i].useWY = useWY;
		threadInfs[i].tau = tau;
		threadInfs[i].ldm = ldm;
		threadInfs[i].b = b;
		threadInfs[i].getTaskMutex = &tMutex;
		threadInfs[i].newTasksCond = &tCond;
		threadInfs[i].condMet = &tCondMet;
		threadInfs[i].taskGrid = taskGrid;
		threadInfs[i].taskM = p;
		threadInfs[i].taskN = q;
		threadInfs[i].taskGrid = taskGrid;
	}

	/* Fix tattr such that created threads can be joined later. */
	pthread_attr_init(&tattr);
	pthread_attr_setdetachstate(&tattr, PTHREAD_CREATE_JOINABLE);
	
	/* Get timing start info. */
	tick = getticks();

	/* Initialise the counter to start threads. */
	t = 0;
	while(t < NUMTHREADS)
	{
		pthread_create( &threads[t], &tattr, pthr_doTasks, (void*) &threadInfs[t] );
		t ++;
	}

	/* Set the starting flag. */
	tCondMet = 1;

	/* Loop through the created threads, finishing execution. */
	t = 0;
	while(t < NUMTHREADS)
	{
		pthread_join(threads[t], NULL);
		t ++;
	}

	/* Get finishing timing data & register result. */
	tock = getticks();
	printf("CPU: %5.2f ms\n", (float)(tock - tick)/3.8e9*1000);

	/* Clean up per-thread working vector. */	
	for(i = 0; i < NUMTHREADS; i ++)
	{
		deleteMatrix( threadInfs[i].wspace[0] );
		deleteMatrix( threadInfs[i].wspace[1] );
	}

	/* Clean up the joinable attribute. */
	pthread_attr_destroy( &tattr );
}

/**
 * \brief Fetches tasks from the task grid until a valid task is returned or there are no more tasks possible
 * \param mutex The mutex locking access to the task grid
 * \param cond The condition to wait to be broadcast, representing a new task available
 * \param t The task to fill the data into
 * \param Grid The grid of tasks
 * \param taskM Number of rows of tasks
 * \param taskN Number of columns of tasks
 * \returns An integer relating to the status of the task in t
 */
int pthr_getNextTask	(pthread_mutex_t* mutex,
			pthread_cond_t* cond,
			int* condMet,
			Task* t,
			Task* Grid,
			int taskM, int taskN)
{
	//initialise return value 
	int ret = TASK_NONE;

	//lock mutex
	pthread_mutex_lock(mutex);

	//repeat until have valid task
	while(1)
	{
		//fetch a task from the grid
		ret = getNextTask(t, Grid, taskM, taskN);
		
		//if no ready tasks
		if(ret == TASK_NONE)
			//wait until tasks are ready
			pthread_cond_wait(cond, mutex);
		else break;//otherwise break with the task
	}
	//release the mutex
	pthread_mutex_unlock(mutex);

	return ret;
}

/**
 * \brief Registers a task as complete in the task grid
 * \param mutex The mutex controlling access to the task grid
 * \param cond The condition to broadcast
 * \param taskgrid The grid of tasks
 * \param tM Rows of tasks
 * \param tN Columns of tasks
 * \param doneTask The information about the completed task
 */
void pthr_doneATask	(pthread_mutex_t* mutex,
			pthread_cond_t* cond,
			int* condMet,
			Task* taskgrid,
			int tM, int tN,
			Task doneTask)
{
	//acquire mutex
	pthread_mutex_lock(mutex);

	//register doneTask as finished in the task grid
	doneATask(taskgrid, tM, tN, doneTask);
	
	//broadcast there might be more tasks now
	doPthrBcast(cond, condMet);

	//release mutex
	pthread_mutex_unlock(mutex);
}

/**
 * \brief Thread code to execute tasks while there are some available then return
 * \param threadinfoptr A pointer to a struct ThreadInfo containing data used by the thread.
 */
void* pthr_doTasks(void* threadinfoptr)
{
	int nextTask;
	float *matData, *matTau, **workingVect;
	int ldm, b, taskM, taskN, useWY;
	Task* taskGrid, toDoTask;

	int *condMet;
	pthread_mutex_t *tMutex;
	pthread_cond_t *tCond;

	struct ThreadInfo localThreadInf;

	//get data in local variables
	localThreadInf = *((struct ThreadInfo*)threadinfoptr);
	nextTask = 1;
	
	matData = localThreadInf.mat;
	matTau = localThreadInf.tau;

	workingVect = localThreadInf.wspace;
	ldm = localThreadInf.ldm;
	b = localThreadInf.b;

	taskGrid = localThreadInf.taskGrid;
	taskM = localThreadInf.taskM;
	taskN = localThreadInf.taskN;

	tMutex = localThreadInf.getTaskMutex;
	tCond = localThreadInf.newTasksCond;
	condMet = localThreadInf.condMet;

	useWY = localThreadInf.useWY;

	//while there are tasks available
	while(nextTask != TASK_DONE)
	{
		//fetch next task
		nextTask = pthr_getNextTask(tMutex, tCond, condMet, &toDoTask, taskGrid, taskM, taskN);

		//execute task if not TASK_NONE or TASK_DONE
		if(nextTask == TASK_AVAIL)
		{
			doATask(toDoTask, matData, matTau, b, ldm, workingVect, useWY);

			//finish task
			pthr_doneATask(tMutex, tCond, condMet, taskGrid, taskM, taskN, toDoTask);
		}
	}
	
	//broadcast to release threads that may be waiting
	doPthrBcast(tCond, condMet);

	//return
	pthread_exit(NULL);
}

void doPthrBcast(pthread_cond_t* cond, int* condmet)
{
	//broadcast cond
	pthread_cond_broadcast(cond);
}

/**
 * \brief Based on the type of task t is, executes t on the matrix mat.
 * \param t A task to execute
 * \param mat The matrix to perform the operation on
 * \param b The block size we are using
 * \param ldm The leading dimension of mat
 * \param colVect A pre-allocated 2b*b array of floats for using in the computation
 */
void doATask	(Task t,
		float* mat,
		float* tau,
		int b, int ldm,
		float** colVect,
		int useWY)
{
	float *blockV, *blockA, *blockB, *blockTau;

	//switch based on the type of task we've got
	switch(t.taskType)
	{
		case QRS:
		{
			blockV = mat + CO((t.k*b),(t.k*b),ldm);
			blockTau = tau + CO((t.k*b),(t.k*b),ldm);
			
			if( useWY )
				SGEQRF(blockV, blockTau, b, b, ldm, colVect[0]);
			else
				qRSingleBlock(blockV, b, b, ldm, colVect[0]);
			
			//printf("qr %d,%d\n", t.k, t.k);
			break;
		}
		case SAPP:
		{
			blockV = mat + CO((t.k*b),(t.k*b),ldm);
			blockA = mat + CO((t.k*b),(t.m*b),ldm);
			blockTau= tau + CO((t.k*b),(t.k*b),ldm);

			if( useWY )
				SLARFT( blockA, blockV, blockTau, b, b, ldm, colVect );
			else
				applySingleBlock(blockA, b, b, ldm, blockV);
			
			//printf("sapp %d,%d %d,%d\n", t.k, t.k, t.k, t.m);
			break;
		}
		case QRD:
		{
			blockA = mat + CO((t.k*b),(t.k*b),ldm);
			blockB = mat + CO((t.l*b),(t.k*b),ldm);
			blockTau = tau + CO((t.l*b),(t.k*b),ldm);

			if( useWY )
				STSQRF(blockA, blockB, blockTau, b, b, b, ldm, colVect[0]);
			else
				qRDoubleBlock(blockA, b, b, blockB, b, ldm, colVect[0]);
			
			//printf("qrd %d,%d %d,%d\n", t.k, t.k, t.l, t.k);
			break;
		}
		case DAPP:
		{
			blockV = mat + CO((t.l*b),(t.k*b),ldm);
			blockA = mat + CO((t.k*b),(t.m*b),ldm);
			blockB = mat + CO((t.l*b),(t.m*b),ldm);
			blockTau = tau + CO((t.l*b),(t.k*b),ldm);

			if( useWY )
				SSSRFT(	blockV,
							blockA, blockB,
							blockTau,
							b, b, ldm);
			else
				applyDoubleBlock(blockA, b, blockB, b, b, ldm, blockV);
			
			//printf("dapp %d,%d %d,%d %d,%d\n", t.l, t.k, t.k, t.m, t.l, t.m);
			break;
		}
	}
}

void updatekthSingleWY	(float* blockV,
			float* tauBlock,
			float beta,
			int k,
			int m, int n, int ldm,
			float* w)
{
	int l, i;
	/* Householder vector is at [0..1,block[j+1:m]] */
	
	/* Compute b = V_(k,:) */
	//for(i = 0; i <= k - 1; i ++)
	//{
	//	w[i] = blockV[(i*ldm) + k];
	//}
	
	/* Compute b = V^T*V_j */
	//for(l = k+1; l < m; l ++)
	//{
	//	for(i = 0; i <= k - 1; i ++)
	//		w[i] += blockV[(k*ldm) + l] * blockV[(i*ldm) + l];
	//}

	/* Compute T_j = T*w */
	//for(l = 0; l <= k - 1; l ++)
	//{
	//	for(i = 0; i <= l; i ++)
	//		tauBlock[(k*ldm) + i] += w[l] * tauBlock[(l*ldm) + i];
	//}
	
	/* Compute T_j = beta * T_j */
	//for(i = 0; i <= k - 1; i ++)
	//	tauBlock[(k*ldm) + i] *= (-beta);

	/* Insert beta on the diagonal of Tau */
	tauBlock[k] = beta;
}

void updateSingleQ_WY	(float* block,
			float* tauBlock,
			int k,
			int m, int n, int ldm,//dims of block
			float* workVector)
{
	/* Compute A = A - 2/v'v*vv'A */
	int i, j;
	float beta = 1.0f, prod;
	
	for(i = k + 1; i < m; i ++)
	{
		beta += workVector[i - k - 1] * workVector[i - k - 1];
	}
	/* Finish computation of 2/v'v */
	beta = (-2)/beta;
	
	for(j = k; j < 32; j ++)
	{
		/* Compute prod = v'A_j */
		prod = block[(j*ldm) + k];//(k,k) to (k,n)

		for(i = k + 1; i < m; i ++)
			prod += block[(j*ldm) + i] * workVector[i - k - 1];

		/* Compute A_j = A_j - beta*v*prod */
		block[(j*ldm) + k] += beta * prod;

		for(i = k + 1; i < m; i ++)
			block[(j*ldm) + i] += beta * prod * workVector[i - k - 1];
	}

	/* Insert nonessential vector below diagonal. */
	for(i = k + 1; i < m; i ++)
		block[(k*ldm) + i] = workVector[i - k - 1];
	
	updatekthSingleWY	(block,
				tauBlock,
				-beta,
				k, m, n, ldm,
				workVector);
}

void SGEQRF	(float* block,
		float* tauBlock,
		int m, int n, int ldm,
		float* workVector)
{
	int k;
	float* xVect;
	
	xVect = block;

	for(k = 0; k < n; k ++)
	{
		/* Get kth householder vector into position starting at workVector */
		calcvkSingle(xVect, m-k, workVector);

		/* Apply householder vector (with an implied 1 in first element to block,
		   generating WY matrices in the process.
		   Stores vector below the diagonal. */
		updateSingleQ_WY	(block, tauBlock,
					k, m, n, ldm,
					workVector);

		/* Shift one along & one down */
		xVect += ldm + 1;
	}
}

void SLARFT	(float* block,
		float* blockV,
		float* tauBlock,
		int m, int n, int ldm,
		float** w)
{
	//applySingleBlock( block, m, n, ldm, blockV );
	/* 	Perform the transformation block = block - blockV*(tauBlock*(blockV^T*block)) 
	 	Equivalent to B = B - V(T(V^TB))
		Noting that T is upper triangular, and V is unit lower triangular. */
	int i, j, k;

	float tau, beta;

	//for(j = 0; j < n; j ++)
	//{
		/* Zero temporary storage. */
	//	for(i = 0; i < m; i ++)
	//	{
	//		w[0][i] = 0;
	//		w[1][i] = 0;
	//	}

		/* Compute w[0] = V^T*b_j */
	//	for(k = 0; k < m; k ++)
	//	{
			/* Compute up to diagonal of V (where i<k). */
	//		for(i = 0; i < k; i ++)
	//		{
	//			w[0][i] += blockV[(i*ldm) + k] * block[(j*ldm) + k];
	//		}

			/* compute where i == k, and V_kk = 1 */
	//		w[0][k] += block[(j*ldm) + k];
	//	}

		/* Compute w[1] = T*w[0] */
	//	for(k = 0; k < m; k ++)
	//	{
			/* Compute above and on diagonal of T. Where i <= k. */
	//		for(i = 0; i <= k; i ++)
	//		{
	//			w[1][i] += tauBlock[(k*ldm) + i] * w[0][k];
	//		}
	//	}
	
		/* Compute b_j = b_j - V*w[1] */
	//	for(k = 0; k < m; k ++)
	//	{
			/* V is one on diagonal. */
	//		block[(j*ldm) +k] -= w[1][k];
			
			/* Compute below diagonal of V. */
	//		for(i = k + 1; i < m; i ++)
	//		{
	//			block[(j*ldm) + i] -= blockV[(k*ldm) + i] * w[1][k];
	//		}
	//	}
	//}

	/* For each column of the block. */
	for(j = 0; j < n; j ++)
	{
		/* Apply successive reflectors with b_j - tau_k*v_k*v_k'b_j */
		for(k = 0; k < n; k ++)
		{
			/* tau_k is at blockV(k) */
			tau = tauBlock[k];
	
			/* Compute v_k'*b_j, with v_k,k = 1 implied */
			beta = block[(j*ldm) + k];//*1.0

			/* Rest of vector. */
			for(i = k+1; i < m; i ++)
				beta += blockV[(k*ldm) + i] * block[(j*ldm) + i];

			beta *= tau;

			/* Compute b_j = b_j - beta*v_k, again with an implied 1 at v_kk */
			block[(j*ldm) + k] -= beta;/* *1.0 */
			
			/* Compute for rest of b_j */
			for(i = k+1; i < m; i ++)
				block[(j*ldm) + i] -= beta * blockV[(k*ldm) + i];
		}
	}
}

void updateDoubleQ_WY	(float* blockA,
			float* blockB,
			float* blockTau,
			int k, int ma, int mb, int n,
			int ldm,
			float* hhVector)//bottom, essential part.
{
	int i, j, m = ma + mb;

	float tau = 1.0, beta;

	/* Compute tau = 2/v'v */
	for(i = 0; i < mb; i ++)
		tau += hhVector[i] * hhVector[i];

	tau = 2/tau;

	for(j = k; j < n; j ++)
	{
		/* Compute v'*b_j */
		beta = blockA[(j*ldm) + k];

		/* Then for lower half */
		for(i = 0; i < mb; i ++)
			beta += blockB[(j*ldm) + i] * hhVector[i];

		beta *= tau;

		/* Compute b_j = b_j - beta*v_k */
		blockA[(j*ldm) + k] -= beta;
		
		for(i = 0; i < mb; i ++)
			blockB[(j*ldm) + i] -= beta * hhVector[i];
	}

	/* Insert vector below diagonal. */
	for(i = 0; i < mb; i ++)
		blockB[(k*ldm) + i] = hhVector[i];

	blockTau[k] = tau;
}

void STSQRF	(float* blockA,
		float* blockB,
		float* blockTau,
		int ma,
		int mb,
		int n,
		int ldm,
		float* hhVector)
{
	int k;
	float* xVectA, *xVectB;
	
	xVectA = blockA;
	xVectB = blockB;

	for(k = 0; k < n; k++)
	{
		//vk = sign(x[1])||x||_2e1 + x
		//vk = vk/vk[0]
		calcvkDouble(xVectA[0], ma - k, xVectB, (ma + mb) - k, hhVector);//returns essential

		//matA(k:ma,k:na) = matA(k:ma,k:na) - (2/(vk.T*vk))*vk*(vk.T*matA(k:ma,k:na)
		//update both blocks, preserving the vectors already stored below the diagonal in the top block and treating them as if they were zeros.
		updateDoubleQ_WY	(blockA, blockB,
					blockTau,
					k, ma, mb, n,
					ldm,
					hhVector + ma - k);

		xVectA += ldm + 1;
		xVectB += ldm;
	}
}

void SSSRFT	(float* blockV,
		float* blockA, float* blockB,
		float* blockTau,
		int b, int n, int ldm)
{
	int i, j, k;

	float tau, beta;

	/* Compute b_j = b_j - tau*v*v'*b_j for each column j of blocks A & B,
	   and for each householder vector v of blockV */

	/* For each column of B */
	for(j = 0; j < n; j ++)
	{
		/* For each householder vector. */
		for(k = 0; k < n; k ++)
		{
			/* tau = 2/v'v, computed earlier, stored in T(k,k). */
			tau = blockTau[k];

			/* Compute beta = v_k'b_j. */
			/* v_k is >0 (=1) only at position k in top half. */
			beta = blockA[(j*ldm) + k];

			/* For lower portion of v_k, aligning with the lower block */
			for(i = 0; i < b; i ++)
				beta += blockB[(j*ldm) + i] * blockV[(k*ldm) + i];

			beta *= tau;
			
			/* Compute b_j = b_j - beta * v */
			/* v_k = 1 at (k) in top half again */
			blockA[(j*ldm) + k] -= beta;

			/* Apply to bottom block. */
			for(i = 0; i < b; i ++)
				blockB[(j*ldm) + i] -= beta * blockV[(k*ldm) + i];
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
 * \param hhVector A pointer to a pre-allocated array for use as a scratchpad to store the householder vector
 *
 * \returns void
 */
void qRSingleBlock	(float* block,
			int m,
			int n,
			int ldb,
			float* hhVector)
{
	int k;
	float* xVect;

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
 * \param hhVector A pre-allocated array for using during the computation
 *
 * \returns void
 */
void qRDoubleBlock	(float* blockA,
			int am,
			int an,
			float* blockB,
			int bm,
			int ldm,
			float* hhVector)
{
	int k;
	float* xVectB, *xVectA;
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
void applySingleBlock	(float* block,
			int m,
			int n,
			int ldb,
			float* hhVectors)
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
void applyDoubleBlock	(float* blockA,
			int am,
			float* blockB,
			int bm,
			int n,
			int ldm,
			float* hhVectorsB)
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

void updateDoubleQ	(float* matA,
			int ma,
			int na,
			float* matB,
			int mb,
			int ldm,
			float* v)
{
	int i, j, k, cols = na;
	float z, a, y;

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

void updateDoubleQZeros	(float* matA,
			int ma,
			int na,
			float* matB,
			int mb,
			int ldm,
			float* v,
			int l)
{
	int i, j, k, rows = l, cols = na;
	float z, a, y;

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
void updateSingleQInp	(float* mat,
			int m,
			int n,
			int ldm,
			float* v)
{
	int i, j, k;
	float z, a, y;

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
void updateSingleQ	(float* mat,
			int m,
			int n,
			int ldm,
			float* v)
{
	int i, j, k;
	float z, a, y;

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
void insSingleHHVector	(float* block,
			int m,
			float* vector)
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
void calcvkSingle	(float* x,
			int l,
			float* vk)
{
	int sign, i;
	float norm, div, beta;

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
void calcvkDouble	(float topDiag,
			int ma,
			float* xb,
			int l,
			float* vk)
{
	int sign, i;
	float norm, div;
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
float do2norm(float* x, int l)
{
	float sum = 0, norm;
	int i;

	for(i = 0; i < l; i++)
		sum += x[i] * x[i];

	norm = sqrt(sum);

	return norm;
}

float* multAB(float* matA, int ma, int na, int lda, float* matB, int nb, int ldb)
{
	float* matC = NULL;
	int i, j, k;
	matC = newMatrix(ma, nb);
	initMatrix(matC, ma, nb, ma, 0);

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

void copyMatrix(float* mat, int m, int n, int ldm, float* copymat)
{
	int i, j;

	for(j = 0; j < n; j ++)
		for(i = 0; i < m; i ++)
			copymat[i + (j*ldm)] = mat[i + (j*ldm)];
}

int checkEqual(float* matA, float* matB, int m, int n, int ldm)
{
	int i, j;
	float diff, epsilon = EPSILON;
	for(i = 0; i < m; i ++)
	{
		for(j = 0; j < n; j ++)
		{
			diff = matA[CO(i,j,ldm)] - matB[CO(i,j,ldm)];
			if(diff > 0)
			{
				if(diff > epsilon)
				{
					printf("(%d,%d): %2.3f /= %2.3f\n", i, j, matA[CO(i,j,ldm)], matB[CO(i,j,ldm)]);
					return 0;
				}
			}
			else
			{
				if(diff < -1*epsilon)
				{
					printf("(%d,%d): %2.3f /= %2.3f\n", i, j, matA[CO(i,j,ldm)], matB[CO(i,j,ldm)]);
					return 0;
				}
			}
		}
	}
	//printf("%5.3f == %5.3f\n", matA[CO(0,0,ldm)], matB[CO(0,0,ldm)]);

	return 1;
}

void printMatrix(float* mat, int m, int n, int ldm)
{
	int r, c;
	putchar('[');
	for(r = 0; r < m; r++)
	{
		for(c = 0; c < n; c++)
		{
			printf(" %2.3f", mat[CO(r,c,ldm)]);
		}
		if(r != m-1)
			putchar(';');
	}
	printf("]\n");
}

void initMatrix(float* mat, int m, int n, int ldm, int mode)
{
	int r, c;

	for(c = 0; c < n; c++)
	{
		for(r = 0; r < m; r++)
		{
			if(mode == ZERO)
				mat[CO(r,c,ldm)] = 0;
			else if(mode == RAND)
				mat[CO(r,c,ldm)] = rand() % 32;
			else if(mode == RANDZO)
				mat[CO(r,c,ldm)] = ((float)(rand() % 201) - 100.0) / 100.0;
			else if(mode == EYE)
				mat[CO(r,c,ldm)] = r == c ? 1 : 0;
		}
	}
}

void deleteMatrix(float* matptr)
{
	free(matptr);
}

float* newMatrix(int m, int n)
{
	float* matptr;
	matptr = malloc(m * n * sizeof(float));
	return matptr;
}
