extern "C" {
#include "../include/gpucalc.h"
}

#include <stdio.h>

#define NUMSTREAMS 128
#define TID threadIdx.x
#define CO(x,y,ldm) (((y)*ldm) + (x))

/* grid scheduler defines */
#define TLOC(x,y) (((y)*M)+(x))
#define tgrid(x,y) taskGrid[TLOC(x,y)]

#define TASK_AVAIL 0
#define TASK_NONE 1
#define TASK_DONE 2

enum Type {QRS, SAPP, QRD, DAPP};
enum Status {READY, DOING, DONE, NONE, NOTASKS};

typedef struct{
	enum Type taskType;
	int l, m, k;
	enum Status taskStatus;
	int mutex;
} Task;


/* cuda_queue defines */
#define cuda_maxqueues 1

/* Timer functions. */
#ifdef TIMERS
    #define TIMER_TIC_ND if ( threadIdx.x == 0 ) tic = clock();
    #define TIMER_TOC_ND(tid) toc = clock(); if ( threadIdx.x == 0 ) atomicAdd( &cuda_timers[tid] , ( toc > tic ) ? (toc - tic) : ( toc + (0xffffffff - tic) ) );
    #define TIMER_TIC clock_t tic; if ( threadIdx.x == 0 ) tic = clock();
    #define TIMER_TOC(tid) clock_t toc = clock(); if ( threadIdx.x == 0 ) atomicAdd( &cuda_timers[tid] , ( toc > tic ) ? (toc - tic) : ( toc + (0xffffffff - tic) ) );
    #define TIMER_TIC2_ND if ( threadIdx.x == 0 ) tic2 = clock();
    #define TIMER_TOC2_ND(tid) toc2 = clock(); if ( threadIdx.x == 0 ) atomicAdd( &cuda_timers[tid] , ( toc2 > tic2 ) ? (toc2 - tic2) : ( toc2 + (0xffffffff - tic2) ) );
    #define TIMER_TIC2 clock_t tic2; if ( threadIdx.x == 0 ) tic2 = clock();
    #define TIMER_TOC2(tid) clock_t toc2 = clock(); if ( threadIdx.x == 0 ) atomicAdd( &cuda_timers[tid] , ( toc2 > tic2 ) ? (toc2 - tic2) : ( toc2 + (0xffffffff - tic2) ) );
#else
    #define TIMER_TIC_ND
    #define TIMER_TOC_ND(tid)
    #define TIMER_TIC
    #define TIMER_TOC(tid)
    #define TIMER_TIC2
    #define TIMER_TOC2(tid)
#endif

/** Struct for a task queue. */
struct queue_cuda {
    /* Indices to the first and last elements. */
    int first, last;

    /* Number of elements in this queue. */
    volatile int count;

    /* Number of elements in the recycled list. */
    volatile int rec_count;

    /* The queue data. */
    volatile int *data;

	volatile int numIn;
};

/** Timers for the cuda parts. */
enum {
    tid_mutex = 0,
    tid_queue,
    tid_gettask,
    tid_memcpy,
    tid_update,
    tid_pack,
    tid_sort,
    tid_pair,
    tid_self,
    tid_potential,
    tid_potential4,
    tid_total,
    tid_count
    };

/* Timers. */
__device__ float cuda_timers[ tid_count ];

/* The per-SM task queues. */
//__device__ struct queue_cuda cuda_queues[ cuda_maxqueues ];
/*__constant__ int cuda_nrqueues;
__constant__ int cuda_queue_size;*/

__device__ __constant__ int cuda_queue_size;
__device__ struct queue_cuda q;
__device__ int taskupdatemutex;


/**
 * @brief Lock a device mutex.
 *
 * @param m The mutex.
 *
 * Loops until the mutex can be set. Note that only one thread
 * can do this at a time, so to synchronize blocks, only a single thread of
 * each block should call it.
 */

__device__ void cuda_mutex_lock ( volatile int *m ) {
    TIMER_TIC
    while ( atomicCAS( (int *) m , 0 , 1 ) != 0 );
    TIMER_TOC( tid_mutex )
    }


/**
 * @brief Attempt to lock a device mutex.
 *
 * @param m The mutex.
 *
 * Try to grab the mutex. Note that only one thread
 * can do this at a time, so to synchronize blocks, only a single thread of
 * each block should call it.
 */

__device__ int cuda_mutex_trylock ( int *m ) {
    TIMER_TIC
    int res = atomicCAS( m , 0 , 1 ) == 0;
    TIMER_TOC( tid_mutex )
    return res;
    }


/**
 * @brief Unlock a device mutex.
 *
 * @param m The mutex.
 *
 * Does not check if the mutex had been locked.
 */

__device__ void cuda_mutex_unlock ( volatile int *m ) {
    TIMER_TIC
    atomicExch( (int *) m , 0 );
    TIMER_TOC( tid_mutex )
    }
    
    
/**
 * @brief Get a task ID from the given queue.
 *
 */
 
__device__ int cuda_queue_gettask ( void ) {

    int ind, tid = -1;
    
    /* Don't even try... */
    if ( q.rec_count == q.count )
        return -1;

    /* Get the index of the next task. */
    ind = atomicAdd( &q.first , 1 );
        
    /* Wrap the index. */
    ind %= cuda_queue_size;

    /* Loop until there is a valid task at that index, getting a task if there is one */
	
    while( q.rec_count < q.count && (tid = q.data[ind]) < 0 );

	if(tid != -1)
		q.data[ind] = -1;

    /* Return the acquired task ID. */
    return tid;
    
    }


/**
 * @brief Put a task onto the given queue.
 *
 * @param tid The task ID to add to the end of the queue.
 */
 
__device__ void cuda_queue_puttask ( int tid ) {

	int ind;

	/* Get the index of the next task. */
	ind = atomicAdd( &q.last , 1 );
	
	/* wrap index */
	ind %= cuda_queue_size;
    
	/* Wait for the slot in the queue to be empty. */
	while( q.data[ind] != -1 );
	
	/* insert the new task ID */
	q.data[ind] = tid;

	atomicAdd( (int *) &q.numIn, 1);	
}
    
    
/**
 * @brief Get a task from the given task queue.
 *
 * @return A valid task ID from the queue or -1 if the queue
 * is empty.
 *
 * Picks tasks from the queue sequentially and checks if they
 * can be computed. If not, they are returned to the queue.
 *
 * This routine blocks until a valid task is picked up, or the
 * specified queue is empty.
 */
 
__device__ int runner_cuda_gettask ( void ) {

    	int tid = -1;
    
    	TIMER_TIC
    
	/* modified because all >0 tasks in queue represent valid
	   tasks in the scheduling structure */
	tid = cuda_queue_gettask();
        
    	/* Put this task into the recycling queue, if needed. */
    	if ( tid >= 0 ) {
            atomicAdd( (int *)&q.rec_count , 1 );
		atomicAdd( (int *)&q.numIn , -1 );
        }
        
    	TIMER_TOC(tid_queue);
        
    	/* Return whatever we got. */
    	return tid;
}

__device__ void init_cuda_queue( int qlen, int totalNumTasks, volatile int *newData)
{
	int j;

	if( ( TID == 0 ) && (blockIdx.x == 0) )
	{
		q.first = 0;
		q.last = 0;
		q.rec_count = 0;
		q.count = totalNumTasks;
		q.data = newData;
		q.numIn = 0;
	}

	for( j = 0; j < qlen; j ++)
	{
		q.data[j] = -1;
	}
}

/* Insert new information into the task structure at (x,y), then place the
   newly revised index into the scheduler queue. */

__device__ void makeTask(volatile Task* taskGrid, int M, int x, int y, enum Type newType, enum Status newStatus, int newK )
{
	if( atomicCAS( (int *) &tgrid(x,y).taskStatus, (int) NONE, (int) READY) == NONE ||
		atomicCAS( (int *) &tgrid(x,y).taskStatus, (int) DONE, (int) READY ) == DONE )
	{
		tgrid(x,y).taskType = newType;
		tgrid(x,y).k = newK;
	
		cuda_queue_puttask( TLOC(x,y) );
	}
}

__device__ void init_cuda_scheduler( volatile Task* taskGrid, int M, int N)
{
	int i , j, ref;

	for(j = 0; j < M; j ++)
	{
		ref = j*M;
		for(i = 0; i < N; i ++)
		{
			taskGrid[ref].l = i;
			taskGrid[ref].m = j;
			taskGrid[ref].taskStatus = NONE;
			taskGrid[ref].k = 0;
			cuda_mutex_unlock(&taskGrid[ref].mutex);
			ref ++;
		}
	}

	makeTask( taskGrid, M, 0, 0, QRS, READY, 0 );
}

__device__ enum Type getNextType(int p, int q, int k)
{
	enum Type ret;
	if(p == k)
	{
		if(q == k)
			ret = QRS;//on diagonal
		else if(q > k)
			ret = SAPP;//on diagonal row
	}
	else if(p > k)
	{
		if(q == k)
			ret = QRD;//on diagonal column
		else if(q > k)
			ret = DAPP;//in the rest
	}

	return ret;
}

__device__ int inGrid(int M, int N, int x, int y)//1 if (x,y) in grid, 0 if not
{
	int ret = 1;

	if (x >= M)
		ret = 0;
	else if (y >= N)
		ret = 0;
	else if (x < 0)
		ret = 0;
	else if (y < 0)
		ret = 0;

	return ret;
}

//checks k equal or greater and done status
__device__ int genericdone(volatile Task* taskGrid, int M, int x, int y, int k)
{
	int ret = 0;

	if(tgrid(x,y).k == k)
	{
		if(tgrid(x,y).taskStatus == DONE)
			ret = 1;
	}
	if(tgrid(x,y).k > k)
		ret = 1;

	return ret;
}

//checks if a sqr has been performed for step k
__device__ int qrsdone(volatile Task* taskGrid, int M, int k)
{
	int ret = 0;

	if(genericdone(taskGrid, M, k, k, k))
	{
		if(tgrid(k,k).taskType == QRS)
			ret = 1;
	}

	return ret;
}

//checks if a dapp has been applied to (x,y) at step k
__device__ int dappdone(volatile Task* taskGrid, int M, int N, int x, int y, int k)
{
	int ret = 0;

	if(!inGrid(M, N, x, y))
		return 1;

	if(genericdone(taskGrid, M, x, y, k))//finished operation
	{
		if(tgrid(x,y).taskType == DAPP)//is dapp task
			ret = 1;
	}

	return ret;
}

//checks if float qr has been performed on (x,y) at step k
__device__ int qrddone(volatile Task* taskGrid, int M, int N, int x, int y, int k)
{
	int ret = 0;
	
	if(genericdone(taskGrid, M, x, y, k))//check if task finished
	{
		if(tgrid(x,y).taskType == QRD)//is qrd task
			ret = 1;
	}
	
	return ret;
}

//can always do qrs if in grid
__device__ int candoQRS(volatile Task* taskGrid, int M, int N, int x, int y, int k)
{
	return inGrid(M, N, x, y);
}

//if can apply at (x,y) step k, returns 1. 0 otherwise
__device__ int candoSAPP(volatile Task* taskGrid, int M, int N, int x, int y, int k)
{
	int ret = 0;

	if (!inGrid(M, N, x - 1, y))
		return 1;

	//checkqrs(k,k)k done, check vectors are ready
	if(qrsdone(taskGrid, M, k))
	{
		//checkdapp(x,y)k-1 done//check previous step completed
		if(dappdone(taskGrid, M, N, x, y, k - 1))
			ret = 1;
	}

	return ret;
}

__device__ int candoQRD(volatile Task* taskGrid, int M, int N, int x, int y, int k)
{
	int ret = 0;
	//checkgendone(x-1,k)k done check if row above is done (qrd or qrs)
	if(genericdone(taskGrid, M, x-1, y, k))
	{
		//checkdapp(x,y)k-1 done check if dapp in place has been done
		if(k == 0)//if no previous
			ret = 1;
		else if(dappdone(taskGrid, M, N, x, y, k-1))
			ret = 1;
	}

	if(!inGrid(M, N, x, y))
		ret = 0;

	return ret;
}

__device__ int candoDAPP(volatile Task* taskGrid, int M, int N, int x, int y, int k)
{
	int ret = 0;
	//checkqrd(x,k)k done
	if(qrddone(taskGrid, M, N, x, k, k))
	{
		//checkgendone(x-1,y)k done
		if(genericdone(taskGrid, M, x-1, y, k))
		{
			//checkdapp(x,y)k-1 done
			if(k == 0)
				ret = 1;
			else if(dappdone(taskGrid, M, N, x, y, k-1))
				ret = 1;
		}
	}

	return ret;
}

/* Register the finished task as completed, then go through the possible 
   successors and add them to the task structure if it is possible to add them */
__device__ void completeATask	(volatile Task* taskGrid,
				int M, int N,
				Task t)
{
	int k, j, p, q;
	enum Type tType, tTypeNext;
	

	p = t.l;
	q = t.m;
	k = tgrid(p,q).k;
	tType = getNextType(p, q, k);

	tgrid(p,q).taskStatus = DONE;
	
	switch(tType)
	{
		case QRS:
		{
			if(candoQRD(taskGrid, M, N, p+1, q, k))//check one below
				makeTask(taskGrid, M, p+1, q, QRD, READY, k);

			for(j = k + 1; j < N; j ++)//check along row
			{
				cuda_mutex_lock(&tgrid(p,j).mutex);

				if(candoSAPP(taskGrid, M, N, p, j, k))
					makeTask(taskGrid, M, p, j, SAPP, READY, k);

				cuda_mutex_unlock(&tgrid(p,j).mutex);
			}
			break;
		}
		case SAPP:
		{
			if(candoDAPP(taskGrid, M, N, p+1, q, k))//check one below
				makeTask(taskGrid, M, p+1, q, DAPP, READY, k);

			break;
		}
		case QRD:
		{
			if(inGrid(M, N, p+1,q))
			{
				cuda_mutex_lock(&tgrid(p+1,q).mutex);
				if(candoQRD(taskGrid, M, N, p+1, q, k))//check one below
					makeTask(taskGrid, M, p+1, q, QRD, READY, k);
				cuda_mutex_unlock(&tgrid(p+1,q).mutex);
			}

			for(j = k + 1; j < N; j ++)
			{
				cuda_mutex_lock(&tgrid(p-1,j).mutex);
				if(candoDAPP(taskGrid, M, N, p, j, k))//check along row
					makeTask(taskGrid, M, p, j, DAPP, READY, k);
				cuda_mutex_unlock(&tgrid(p-1,j).mutex);
			}
			
			break;
		}
		case DAPP:
		{
			tTypeNext = getNextType(p, q, k + 1);

			switch(tTypeNext)//check whether can activate any for next step
			{
				case QRS:
				{
					if(candoQRS(taskGrid, M, N, p, q, k + 1))
						makeTask(taskGrid, M, p, q, QRS, READY, k + 1);
					break;
				}
				case SAPP:
				{
					if(candoSAPP(taskGrid, M, N, p, q, k + 1))
						makeTask(taskGrid, M, p, q, SAPP, READY, k + 1);

					break;
				}
				case QRD:
				{
					if(candoQRD(taskGrid, M, N, p, q, k + 1))
						makeTask(taskGrid, M, p, q, QRD, READY, k + 1);

					break;
				}
				case DAPP:
				{
					if(candoDAPP(taskGrid, M, N, p, q, k + 1))
						makeTask(taskGrid, M, p, q, DAPP, READY, k + 1);
					break;
				}
			}

			if(candoDAPP(taskGrid, M, N, p + 1, q, k))//check one below in current step
				makeTask(taskGrid, M, p + 1, q, DAPP, READY, k);

			break;
		}
	}
}

//computes the sum of the elements of the size startN sumVector, storing the result in sumVector[0] in 5 cycles
__device__ void reduceSum(	volatile float* sumVector,
				char startN)
{
	char n = startN >> 1;

	while(n > 0)
	{
		if(TID < n)
			sumVector[TID] = sumVector[TID] + sumVector[TID + n];
		n = n >> 1;//n /= 2
	}
}	

__device__ void calcHH(	float matelem,//the block containing a column to calculate the householder vector of
			volatile float hhVector[],//the array to store the resulting vector in
			int k)//the step at which this was called
{
	float localdiv;
	int sign;

	//read vectors in from below the diagonal(k)
	if(TID >= k)
		hhVector[TID] = matelem;
	if(TID < k)//zero above diagonal
		hhVector[TID] = 0;

	//square each element
	hhVector[TID] *= hhVector[TID];
	
	//ideally only do required computation here; not necessarily 16, 8, 4, 2, 1
	//reduction to calculate the sum of squares of the 32 element vector
	reduceSum(hhVector, 32);

	//calculate sign*norm and put in local variable
	localdiv = sqrt(hhVector[0]);

	//if norm not 0
	if(localdiv != 0.0)
	{
		hhVector[TID] = matelem;

		sign = hhVector[k] >= 0 ? 1 : -1;
		
		localdiv *= sign;
		//add element in block at (k,k) to norm and store in vector
		localdiv += hhVector[k];

		//divide because want v(k+1:m) = v(k+1:m)/v(k) = v(k+1:m) * (1/v(k))
		localdiv = 1.0/localdiv;
	}
	else//if norm is zero, 
		localdiv = 1.0;

	if(TID < k)
		hhVector[TID] = 0;
	if(TID == k)
		hhVector[TID] = 1;
	if(TID > k)
		hhVector[TID] = matelem * localdiv;
}

__device__ void applyHH(float blockRow[],// the 32*32 matrix block to compute over
			volatile float workingVector[],//SHARED a working space of size 32 used in the computation
			int k,//the step at which the function was called, A(k:m, k:n) is submatrix to operate on
			float hhVectorelem)//element of vector storing the householder vector, zero above diagonal
{
	float 	y,//y will be -2 divided by the sum of squares of vector
		z;//z will be v[TID] * sum(A(:,j) .* v) * y

	int 	j;//column counter, reference in block

	//thread TID starts at (TID, k)
	//blockref = (k*32) + TID;
	
	//read data for summation
	workingVector[TID] = hhVectorelem;
	//square elements of working vector
	workingVector[TID] *= workingVector[TID];

	//reduction to find sum of squares of workingVector
	reduceSum(workingVector, 32);

	//make y a register equal to -2/sum of squares of v(v'*v)
	y = (-2.0) / workingVector[0];
	
	for(j = k; j < 32; j ++)//submatrix A(k:m,k:n)
	{
		//fill workingVector with the componentwise multiplication of householder vector (zero above k) and column j of block
		//workingVector[TID] = matblock[blockref] * hhVectorelem;
		workingVector[TID] = blockRow[j] * hhVectorelem;

		//reduction to find sum of multiplication of column of block with hhVector
		reduceSum(workingVector, 32);
		
		//set z = TIDth element of v times -2/v'v
		z = hhVectorelem * y;

		//multiply z by sum of componentwise multiplication of column with hhVector in workingVector[0]
		z *= workingVector[0];
		
		//add z to block(TID,j). zero above diagonal
		blockRow[j] += z;

		//row major storage, next column is 32 elements ahead
		//blockref += 32;
	}
	
	if(TID > k)//store essential part of vector below diagonal
		blockRow[k] = hhVectorelem;//insert essential part of vector below diagonal in column k of block
}

__device__ void calcDoubleHH	(float topElem,//element of top block
				float lowElem,
				volatile float hhVector[], //SHARED 32x1 array to insert vector
				int k)//step at which called. use column k
{
	float tmp;//elemK not used in TID > 0
	int sign;

	if(TID < k)//zero above diagonal in top
		hhVector[TID] = 0.0;

	if(TID == k)//top nonzero element
		hhVector[k] = topElem;

	if(TID > k)//zero below diagonal in top
		hhVector[TID] = 0.0;

	//all read lower block in
	hhVector[TID + 32] = lowElem;

	if(TID == k)//kth thread holds non zero element in top block
	{
		sign = topElem >= 0 ? 1 : -1;
	}

	//all threads hold elements from bottom block
	tmp = hhVector[TID + 32];

	//square top nonzero in hhVector
	if(TID == k)
		hhVector[k] *= hhVector[k];

	//square each element in bottom block
	hhVector[TID + 32] *= hhVector[TID + 32];

	//reduce to compute sum of squares in 0th element of 64 element hhVector
	reduceSum(hhVector, 64);

	if(TID == k)
	{
		//store sign * norm in kth position
		hhVector[k] = sign * sqrt(hhVector[0]);

		if(hhVector[k] != 0.0)
		{
			//add sign*norm to kth element and store
			hhVector[k] = topElem + hhVector[k];
			
			//divide because want to divide by hhVector[k]
			hhVector[k] = 1.0/hhVector[k];
		}
		else//norm zero
			hhVector[k] = 1.0;
	}

	//normalise by multiplying by kth element
	if(TID != k)
		hhVector[32 + TID] = tmp * hhVector[k];
	if(TID == k)
		hhVector[32 + k] = tmp * hhVector[k];

	if(TID == k)//top part is 1 on diagonal
		hhVector[k] = 1.0;
	else if(TID != k)//zero elsewhere
		hhVector[TID] = 0.0;
}

__device__ void applyDoubleHH	(float topRow[],
				float lowRow[],
				volatile float workingVector[],
				int k,
				float hhVectorElem)
{
	float	y,//-2/v'v
		zupp, zlow;//y * v[i] *sum(A(:,j) .* v) for both blocks

	int 	j;//column counter

	//copy hhVector and square square each element for summation
	if(TID == k)
		workingVector[TID] = 1.0;
	if(TID != k)
		workingVector[TID] = 0.0;

	workingVector[TID + 32] = hhVectorElem * hhVectorElem;
	
	//reduce to find sum of squares
	reduceSum(workingVector, 64);
	
	//set y = -2/sum of squares
	y = (-2) / workingVector[0];

	//for each column
	for(j = k; j < 32; j ++)
	{
		//fill working vector[i] with top block(i,j) * hhVector[i]
		if(TID == k)
			workingVector[TID] = topRow[j];
		if(TID != k)
			workingVector[TID] = 0.0;

		//fill workingVector[i + 32] with bottom block(i,j) * hhVector[i+32]
		workingVector[TID + 32] = lowRow[j] * hhVectorElem;

		//sum to find sum of componentwise multiplication
		reduceSum(workingVector, 64);
		
		//set zupp = TIDth element of hhvector times -2/v'v
		if(TID == k)
			zupp = y;
		if(TID != k)
			zupp = 0.0;

		zlow = y * hhVectorElem;

		//multiply both by sum of multiplication
		zupp *= workingVector[0];
		zlow *= workingVector[0];
		
		//add to top block element
		topRow[j] += zupp;

		//add to bottom block element
		lowRow[j] += zlow;
	}
	
	lowRow[k] = hhVectorElem;
}


__device__ void applyHHPrime	(float velem,
				float applyRow[],
				int k,
				volatile float workingVector[])
{
	//apply the vector starting at row k+1 of hhVector to the (k:32) portion of applyVector
	float y, z;
	
	int j;

	//load workingVector with squares of hhVector
	workingVector[TID] = velem * velem;

	//sum to find sum of squares of hhVector
	reduceSum(workingVector, 32);

	//set y = -2/sumsquares
	y = (-2) / workingVector[0];

	//apply to columns 0:32
	for(j = 0; j < 32; j ++)
	{
		//load working vector with componentwise multiplication of hhVector with column of block A
		workingVector[TID] = velem * applyRow[j];

		//sum to find sum of pairwise mult
		reduceSum(workingVector, 32);

		//set z = sum of this times -2/sumvector
		z = y * velem;

		//multiply by element of hhVector
		z *= workingVector[0];
	
		//add to applyVector in place
		applyRow[j] += z;
	}
}

__device__ void applyDoubleHHPrime	(float velem,
					float topAppRow[],
					float lowAppRow[],
					int k,
					volatile float workingVector[])
{
	float y, z;

	int j;

	//load implied top vector
	if(TID == k)
		workingVector[k] = 1.0;
	
	if(TID != k)
		workingVector[TID] = 0.0;
	
	//square lower elements (top is 001000 so no need to multiply)
	workingVector[TID + 32] = velem * velem;

	//reduce to find sum of squares
	reduceSum(workingVector, 64);

	//set y = -2/sum of squares
	y = (-2) / workingVector[0];

	//apply to columns (0:32)
	for(j = 0; j < 32; j ++)
	{
		//load workingVector with 1 * *topAppElem (want workingVector to have component mult of hhvector and elements of application block
		if(TID == k) 
			workingVector[k] = topAppRow[j];

		if(TID != k)
			workingVector[TID] = 0.0;
	
		//do the same for the lower block but nonzero here
		workingVector[TID + 32] = velem * lowAppRow[j];

		//reduce to find sum of multiplication
		reduceSum(workingVector, 64);

		//set zTop and zLow equal to this sum times -2/v'v
		z = workingVector[0] * y;
	
		//multiply top by sum(mult) * -2/v'v. (velem here "is" 1)
		if(TID == k)
			topAppRow[j] += z;

		//multiply zLow by lower element of hhVector
		z *= velem;

		//add result to lower application element
		lowAppRow[j] += z;
	}
}
__device__ void device_doQRS	( float* matrix, int ldm,
				volatile float workingVector[],
				float bRow[])
{
	int k, j, ref;

	ref = TID;

	for(j = 0; j < 32; j ++)//load row of block into local
	{
		bRow[j] = matrix[ref];
		ref += ldm;
	}

	for(k = 0; k < 32; k ++)
	{
		//calculate the kth hh vector from the kth column of the TIDth row of the matrix
		calcHH	(bRow[k],
			workingVector,
			k);

		//calculate the application of the hhvector along row TID
		applyHH	(bRow,
			workingVector,
			k,
			workingVector[TID]);
	}

	//copy row back
	ref = TID;
	for(j = 0; j < 32; j ++)
	{
		matrix[ref] = bRow[j];
		ref += ldm;
	}
}
__global__ void doQRS( float* matrix, int ldm)
{
	__shared__ volatile float workingVector[32];
	
	float row[32];

	device_doQRS	(matrix, ldm,
			workingVector,
			row);
}

__device__ void device_doQRD	( float* blockA,  float* blockB, int ldm,
				volatile float workingVector[],
				float topRow[],
				float lowRow[])
{
	int k, j, ref;
	
	ref = TID;
	for(j = 0; j < 32; j ++)//for each column
	{
		//read top block
		topRow[j] = blockA[ref];

		//read lower block into lower 32x32 square
		lowRow[j] = blockB[ref];
		ref += ldm;
	}

	for(k = 0; k < 32; k ++)
	{
		//calculate and store the vector
		calcDoubleHH	(topRow[k],
				lowRow[k],
				workingVector,
				k);

		//apply vector to both tidth rows of the matrix
		applyDoubleHH	(topRow,
				lowRow,
				workingVector,
				k,
				workingVector[TID + 32]);
	}

	ref = TID;
	for(j = 0; j < 32; j ++)
	{
		//write back to correct blocks
		blockA[ref] = topRow[j];
		blockB[ref] = lowRow[j];
		ref += ldm;
	}
}

__global__ void doQRD( float* blockA,  float* blockB, int ldm)
{
	__shared__ volatile float workingVector[64];
	
	float rowA[32], rowB[32];

	device_doQRD	(blockA, blockB, ldm,
			workingVector,
			rowA,
			rowB);
}

__device__ void device_doSAPP	( float* blockV,
				 float* blockA,
				int ldm,
				volatile float workingVector[],
				float aRow[],
				float vRow[])
{
	float 	scal,
		mult;

	int j, l, ref;

	ref = TID;
	for(j = 0; j < 32; j ++)
	{
		aRow[j] = blockA[ref];

		if(TID < j)
			vRow[j] = 0.0;
		else if(TID == j)
			vRow[j] = 1.0;
		else if(TID > j)
			vRow[j] = blockV[ref];
		ref += ldm;
	}

	for(l = 0; l < 32; l ++)
	{
		//load vector, ready for reduction
		workingVector[TID] = vRow[l] * vRow[l];

		//compute scal <-- -2/sum(v[tid]^2)
		reduceSum(workingVector, 32);

		scal = (-2) / workingVector[0];

		for(j = 0; j < 32; j ++)
		{
			//sumMult <-- a_tid,j*v_tid,l
			workingVector[TID] = aRow[j] * vRow[l];

			//find for lower elements
			reduceSum(workingVector, 32);

			mult = workingVector[0] * scal;

			//a_tid,j <-- a_tid,j + scal * sumMult * v_tid,l
			aRow[j] += vRow[l] * mult;
		}
	}

	ref = TID;

	for(j = 0; j < 32; j ++)
	{
		blockA[ref] = aRow[j];
		ref += ldm;
	}
}

__global__ void doSAPP	( float* blockV,
			 float* blockA,
			int ldm)
{
	__shared__ volatile float workingVector[32];
	
	float aRow[32], vRow[32];

	device_doSAPP	(blockV, blockA, ldm,
			workingVector,
			aRow,
			vRow);
}

__device__ void device_doDAPP	( float* blockV,
				 float* blockA,
				 float* blockB,
				int ldm,
				volatile float workingVector[],
				float aRow[],
				float bRow[])
{
	float 	vElem,
		scal,
		mult;

	int l, j, ref;

	
	ref = TID;
	for(j = 0; j < 32; j ++)
	{
		aRow[j] = blockA[ref];
		bRow[j] = blockB[ref];
		ref += ldm;
	}

	ref = TID;
	for(l = 0; l < 32; l ++)
	{
		//load vector, ready for reduction
		vElem = blockV[ref];
		workingVector[TID] = vElem * vElem;

		//compute scal <-- -2/sum(v[tid]^2)
		//perform sum on lower half (top is I)
		reduceSum(workingVector, 32);

		scal = (-2) / (workingVector[0] + 1);

		for(j = 0; j < 32; j ++)
		{
			//sumMult <-- a_tid,j*v_tid,l
			workingVector[TID] = bRow[j] * vElem;

			if(TID == l)
				workingVector[TID] += aRow[j];

			//find for lower elements
			reduceSum(workingVector, 32);

			mult = workingVector[0] * scal;

			//a_tid,j <-- a_tid,j + scal * sumMult * v_tid,l

			if(TID == l)
				aRow[j] += mult;

			bRow[j] += mult * vElem;
		}
		ref += ldm;
	}

	ref = TID;
	for(j = 0; j < 32; j ++)
	{
		blockA[ref] = aRow[j];
		blockB[ref] = bRow[j];
		ref += ldm;
	}
}

__global__ void doDAPP	( float* blockV,
			 float* blockA,
			 float* blockB,
			int ldm)
{
	__shared__ volatile float workingVector[32];

	float 	aRow[32],
		bRow[32];

	device_doDAPP	(blockV,
			blockA,
			blockB,
			ldm,
			workingVector,
			aRow,
			bRow);
}

/* Fetch and return a task's information from the task structure, at location
   given by ref. */
inline __device__ void retrieveTask( Task* ret, volatile Task* t )
{
	//__threadfence();

	//read and return value
	t->taskStatus = DOING;
	//printf("tasktzp: %p: %d\n", &ret->l, ret->l);
	ret->l = t->l;
	ret->m = t->m;
	ret->k = t->k;
	ret->taskStatus = DOING;
	ret->taskType = t->taskType;
}

__device__ int executeTask	( Task t,
				float* mat, int ldm, int n,
				volatile float workingVector[],
				float vElems[], float topRow[], float lowRow[])
{
	 float *blockV, *blockA, *blockB;

	//switch based on the type of task we've got
	switch(t.taskType)
	{
		case QRS:
		{
			blockV = mat + CO(t.k*32,t.k*32,ldm);
			device_doQRS	( blockV, ldm,
					workingVector, topRow );
//			if(TID == 0)printf("%d: QRS at %d,%d\n", blockIdx.x, t.k, t.k);
			break;
		}
		case SAPP:
		{
			blockV = mat + CO(t.k*32,t.k*32,ldm);
			blockA = mat + CO(t.k*32,t.m*32,ldm);
			device_doSAPP	( blockV, blockA, ldm,
					workingVector, vElems, topRow );
//			if(TID == 0)printf("%d: SAPP from %d,%d to %d,%d\n", blockIdx.x, t.k, t.k, t.k, t.m);
			break;
		}
		case QRD:
		{
			blockA = mat + CO(t.k*32,t.k*32,ldm);
			blockB = mat + CO(t.l*32,t.k*32,ldm);
			device_doQRD	( blockA, blockB, ldm,
					workingVector, topRow, lowRow );
//			if(TID == 0)printf("%d: QRD on %d,%d; %d,%d\n", blockIdx.x, t.k, t.k, t.l, t.k);
			break;
		}
		case DAPP:
		{
			blockV = mat + CO(t.l*32,t.k*32,ldm);
			blockA = mat + CO(t.k*32,t.m*32,ldm);
			blockB = mat + CO(t.l*32,t.m*32,ldm);
			device_doDAPP	(blockV, blockA, blockB, ldm,
					workingVector, topRow, lowRow );
//			if(TID == 0)printf("%d: DAPP from %d,%d to %d,%d; %d,%d\n", blockIdx.x, t.l, t.k, t.k, t.m, t.l, t.m);

			break;
		}
	}

	return 1;
}

__global__ void taskKernel	(  float* matrix, int m, int n,
				int totTasks,
				volatile Task* taskGrid,
				int M, int N )
{
	__shared__ volatile float workVector[64];
	
	float 	vElems[32],
		topRow[32],
		lowRow[32];

	int taskid;
	Task tasktz;
	__shared__ Task task;
	__shared__ int s_tid;

	/* repeat while there are still tasks undone */
	while(q.rec_count < totTasks )
	{
		/* retrieve task from the cuda queue */
		if(TID == 0)
		{
			taskid = runner_cuda_gettask();
			s_tid = taskid;
			//printf("%d: got task %d\n", blockIdx.x, taskid);
		}

		/* have finished if taskid is less than 0. Might also have invalid task */
		if(s_tid < 0)
		{
			//if(TID == 0)printf("missed ");
			if(q.rec_count < totTasks)asm("trap;");
			continue;
		}

		/* get the specifics of this task from the main task structure */
		if( TID == 0 )
		{
			//printf("tasktzp: %p: %d\n", &tasktz.l, tasktz.l);
			
			retrieveTask( &tasktz, taskGrid + taskid );
			task = tasktz;
		}

		//__threadfence();
		/* perform the activity specified by the task t */
		executeTask	(task, matrix, m, n,
				workVector,
				vElems,
				topRow, lowRow);

		/* register task as finished in the task structure 
		At the same time, insert each newly activated task into the cuda queue */
		//__threadfence();
		if( TID == 0 )
		{
			completeATask( taskGrid, M, N, task );
		}
	}
	//if( (TID == 0) && (blockIdx.x == 0) )	printf("%d, %d\n", q.rec_count, q.numIn);
}

__global__ void cuda_initScheduling	(volatile Task* taskGrid, int p, int q,
					volatile int *newData,
					int qlen, int totTasks)
{
	init_cuda_queue( qlen, totTasks, newData );

	init_cuda_scheduler( taskGrid, p, q );
}

int calcTotalTasks(int m, int n)
{
	int ret;

	//calculate (3n^2m - n^3 + 3mn + n)/6
	ret = n;
	ret += 3*m*n;
	ret -= n*n*n;
	ret += 3*n*n*m;
	ret /= 6;

	//printf("%d,%d %d tasks\n",m, n, ret);
	return ret;
}

extern "C"
void cudaQRTask(float* mat, int m, int n)
{
	int totalTasks, p = m/32, q = n/32, queuelen = p * q;
	volatile int *dev_data;
	//initialise task structure on GPU
	volatile Task* dev_taskGrid;

	enum cudaError cuerr;

	float *dev_m;
	
	totalTasks = calcTotalTasks( p, q );
	//queuelen = totalTasks + 5;

	cuerr = cudaMalloc( (void**) &dev_taskGrid, p * q * sizeof(Task) );
	if(cuerr != cudaSuccess)
		printf("error allocating task grid\n");		
	cuerr = cudaMalloc( (void**) &dev_data, queuelen * sizeof(int) );
	if(cuerr != cudaSuccess)
		printf("error allocating dev queue data\n");
	cuerr = cudaMalloc( (void**) &dev_m, m*n*sizeof(float));
	if(cuerr != cudaSuccess)
		printf("error allocating dev mat\n");

	cuerr = cudaMemcpy(dev_m, mat, m*n*sizeof(float), cudaMemcpyHostToDevice);
	if(cuerr != cudaSuccess)
		printf("error cpying dev mat\n");

	cuerr = cudaMemcpyToSymbol( cuda_queue_size, &queuelen, sizeof(int), 0, cudaMemcpyHostToDevice );
	if(cuerr != cudaSuccess)
		printf("error cpying size\n");

	/* initialise all structures for scheduling operations on the GPU */
	cuda_initScheduling<<<1,1>>>( 	dev_taskGrid, p, q,
					dev_data, queuelen, totalTasks );
	taskKernel<<<128 < p*q ? 128 : p*q -1,32>>>( dev_m, m, n, totalTasks, dev_taskGrid, p, q );

	cuerr = cudaMemcpy(mat, dev_m, m*n*sizeof(float), cudaMemcpyDeviceToHost);
	if(cuerr != cudaSuccess)
		printf("copy back failed %d\n", cuerr);

	cuerr = cudaFree(dev_m);
	if( cuerr != cudaSuccess)
		printf("error freeing m %d\n", cuerr);
	cudaFree((Task *) dev_taskGrid);
	if( cuerr != cudaSuccess)
		printf("error freeing task grid %d\n", cuerr);
	cudaFree((int *) dev_data);
	if( cuerr != cudaSuccess)
		printf("error freeing matrix %d\n\n", cuerr);

	if(cuerr != cudaSuccess)
		cudaDeviceReset();//printf("error freeing m %d\n", cuerr);
}

extern "C"
void cudaQRFull(float* mat, int m, int n)
{
	int i, j, k, p, q, s;
	int blockdm;

	float* dev_m, *dev_K, *dev_V, *dev_A, *dev_B;

	cudaStream_t streams[NUMSTREAMS];
	
	for(k = 0; k < NUMSTREAMS; k ++)
		cudaStreamCreate(&streams[k]);

	p = m/32;
	q = n/32;

	blockdm = 32*m;//block to block dim along row

	cudaMalloc((void**) &dev_m, m*n*sizeof(float));
	cudaMemcpy(dev_m, mat, m*n*sizeof(float), cudaMemcpyHostToDevice);

	dev_K = dev_m;

	for(k = 0; k < q; k ++)
	{
		doQRS<<<1, 32, 0, streams[0]>>>(dev_K, m);
		cudaDeviceSynchronize();

		s = 1;

		dev_A = dev_K + blockdm;//one along
		for(j = k+1; j < q; j ++)
		{
			doSAPP<<<1, 32, 0, streams[s]>>>(dev_K, dev_A, m);
			
			dev_A += blockdm;//advance along row

			s ++;
			s = s % (NUMSTREAMS - 1);
		}

		dev_V = dev_K + 32;//one down from K

		for(i = k+1; i < p; i ++)
		{
			doQRD<<<1, 32, 0, streams[0]>>>(dev_K, dev_V, m);
			cudaDeviceSynchronize();

			s = 0;

			dev_A = dev_K + blockdm;//one along from K			
			dev_B = dev_V + blockdm;//one along from V

			for(j = k+1; j < q; j ++)
			{
				doDAPP<<<1, 32, 0, streams[s]>>>(dev_V, dev_A, dev_B, m);
				dev_A += blockdm;
				dev_B += blockdm;
				
				s ++;
				s = s % NUMSTREAMS;
			}
			dev_V += 32;//one down from previous
		}
		dev_K += blockdm + 32;//one along, one down
		cudaDeviceSynchronize();
	}

	cudaMemcpy(mat, dev_m, m*n*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dev_m);
	for(k = 0; k < NUMSTREAMS; k ++)
		cudaStreamDestroy(streams[k]);
}
