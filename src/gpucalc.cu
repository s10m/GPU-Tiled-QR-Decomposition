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

#define MXBLOCKS 128
#define WARPS 4
/* WARPS MUST be factor of blocksize. */

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

#define TIMERS

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
    tid_waitfortask,
    tid_working,
    tid_doingQRS,
    tid_doingSAPP,
    tid_doingQRD,
    tid_doingDAPP,
    tid_gettingtasks,
    tid_taskinfo,
    tid_executing,
    tid_completing,
    tid_first,
    tid_total,
    tid_count
    };

/* Timers. */
__device__ float cuda_timers [ tid_count ];

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
	//TIMER_TIC
    while( q.rec_count < q.count && (tid = q.data[ind]) < 0 );
	//TIMER_TOC(tid_waitfortask)
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
	cuda_mutex_lock(&tgrid(x,y).mutex);
	if( atomicCAS( (int *) &tgrid(x,y).taskStatus, (int) NONE, (int) READY) == NONE ||
		atomicCAS( (int *) &tgrid(x,y).taskStatus, (int) DONE, (int) READY ) == DONE )
	{
		tgrid(x,y).taskType = newType;
		tgrid(x,y).k = newK;
	
		cuda_queue_puttask( TLOC(x,y) );
	}
	cuda_mutex_unlock(&tgrid(x,y).mutex);
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
	
	cuda_mutex_lock(&tgrid(k,k).mutex);
	if(genericdone(taskGrid, M, k, k, k))
	{
		if(tgrid(k,k).taskType == QRS)
			ret = 1;
	}
	cuda_mutex_unlock(&tgrid(k,k).mutex);

	return ret;
}

//checks if a dapp has been applied to (x,y) at step k
__device__ int dappdone(volatile Task* taskGrid, int M, int N, int x, int y, int k)
{
	int ret = 0;

	if(!inGrid(M, N, x, y))
		return 1;

	cuda_mutex_lock(&tgrid(x,y).mutex);
	if(genericdone(taskGrid, M, x, y, k))//finished operation
	{
		if(tgrid(x,y).taskType == DAPP)//is dapp task
			ret = 1;
	}
	cuda_mutex_unlock(&tgrid(x,y).mutex);

	return ret;
}

//checks if float qr has been performed on (x,y) at step k
__device__ int qrddone(volatile Task* taskGrid, int M, int N, int x, int y, int k)
{
	int ret = 0;
	
	cuda_mutex_lock(&tgrid(x,y).mutex);
	if(genericdone(taskGrid, M, x, y, k))//check if task finished
	{
		if(tgrid(x,y).taskType == QRD)//is qrd task
			ret = 1;
	}
	cuda_mutex_unlock(&tgrid(x,y).mutex);
	
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
	cuda_mutex_lock(&tgrid(x-1,y).mutex);
	//checkgendone(x-1,k)k done check if row above is done (qrd or qrs)
	if(genericdone(taskGrid, M, x-1, y, k))
	{
		//checkdapp(x,y)k-1 done check if dapp in place has been done
		if(k == 0)//if no previous
			ret = 1;
		else if(dappdone(taskGrid, M, N, x, y, k-1))
			ret = 1;
	}
	cuda_mutex_unlock(&tgrid(x-1,y).mutex);

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
		cuda_mutex_lock(&tgrid(x-1,y).mutex);
		if(genericdone(taskGrid, M, x-1, y, k))
		{
			//checkdapp(x,y)k-1 done
			if(k == 0)
				ret = 1;
			else
			{
				if(dappdone(taskGrid, M, N, x, y, k-1))
					ret = 1;
			}
		}
		cuda_mutex_unlock(&tgrid(x-1,y).mutex);
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
	
	cuda_mutex_lock(&tgrid(p,q).mutex);
	tgrid(p,q).taskStatus = DONE;
	cuda_mutex_unlock(&tgrid(p,q).mutex);
	
	switch(tType)
	{
		case QRS:
		{
			if(candoQRD(taskGrid, M, N, p+1, q, k))//check one below
				makeTask(taskGrid, M, p+1, q, QRD, READY, k);

			for(j = k + 1; j < N; j ++)//check along row
			{
				if(candoSAPP(taskGrid, M, N, p, j, k))
					makeTask(taskGrid, M, p, j, SAPP, READY, k);
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
				if(candoQRD(taskGrid, M, N, p+1, q, k))//check one below
					makeTask(taskGrid, M, p+1, q, QRD, READY, k);
			}

			for(j = k + 1; j < N; j ++)
			{
				if(candoDAPP(taskGrid, M, N, p, j, k))//check along row
					makeTask(taskGrid, M, p, j, DAPP, READY, k);
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

__device__ void applyHH(volatile float blockCache[],//SHARED the 32*32 matrix block to compute over
			float* matTau, int ldm,
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
	y = 2.0 / workingVector[0];
	
	for(j = k; j < 32; j ++)//submatrix A(k:m,k:n)
	{
		//fill workingVector with the componentwise multiplication of householder vector (zero above k) and column j of block
		workingVector[TID] = blockCache[TID+(j*32)] * hhVectorelem;

		//reduction to find sum of multiplication of column of block with hhVector
		reduceSum(workingVector, 32);
		
		//set z = TIDth element of v times -2/v'v
		z = hhVectorelem * y;

		//multiply z by sum of componentwise multiplication of column with hhVector in workingVector[0]
		z *= workingVector[0];
		
		//add z to block(TID,j). zero above diagonal
		blockCache[TID + (j*32)] -= z;
	}
	
	if(TID > k)//store essential part of vector below diagonal
		blockCache[TID + k*32] = hhVectorelem;//insert essential part of vector below diagonal in column k of block
	
	if(TID == 0)
		matTau[k] = y;
}

__device__ void calcDoubleHHWY	(float topElem,
				float lowElem,
				int k,
				volatile float blockCache[])
{
	/* Calculate v_k. */
	int sign, i;

	float 	topSum = 0,
		lowSum = 0,
		alpha;

	__shared__ volatile float first;
	
	if(TID == k)
		first = topElem;
	
	topSum = first * first;
	/* Sum squares of V to compute norm, using column of BlockCache as working space. */
	blockCache[TID + (k*32)] = lowElem * lowElem;
	for(i = 0; i < 32; i ++)
		lowSum += blockCache[i + (k*32)];
	
	alpha = sqrt(topSum + lowSum);
	sign = first >= 0 ? 1 : -1;

	if(alpha != 0.0)
	{
		/* Zeroth element is = first + sign*norm */
		alpha = first + sign * alpha;
		alpha = 1.0/alpha;
	}
	else
		alpha = 1.0;

	//topElem *= alpha;
	lowElem *= alpha;
	
	blockCache[TID + (k*32)] = lowElem;
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
				float* blockTau, int ldm,
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
	y = 2 / workingVector[0];

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
		topRow[j] -= zupp;

		//add to bottom block element
		lowRow[j] -= zlow;
	}
	
	if(TID == 0)
		blockTau[k] = y;
	
	lowRow[k] = hhVectorElem;
}

__device__ void device_doQRS	( float* matrix, float* matTau,
				int ldm,
				volatile float workingVector[],
				volatile float blockCache[])
{
	int k, j, refCache, refMat;

	refMat = TID;
	refCache = TID;

	for(j = 0; j < 32; j ++)//load block into shared memory
	{
		blockCache[refCache] = matrix[refMat];

		refMat += ldm;
		refCache += 32;
	}

	for(k = 0; k < 32; k ++)
	{
		//calculate the kth hh vector from the kth column of the TIDth row of the matrix
		calcHH	(blockCache[TID + (k*32)],//(tid, k)
			workingVector,
			k);

		//calculate the application of the hhvector along row TID
		applyHH	(blockCache,
			matTau,
			ldm,
			workingVector,
			k,
			workingVector[TID]);
	}

	//copy row back
	refMat = TID;
	refCache = TID;

	for(j = 0; j < 32; j ++)
	{
		matrix[refMat] = blockCache[refCache];

		refMat += ldm;
		refCache += 32;
	}
	__threadfence();
}

__global__ void doQRS( float* matrix, float* tau, int ldm)
{
	__shared__ volatile float workingVector[32];
	__shared__ volatile float blockCache[32*32];
	
	device_doQRS	(matrix, tau,
			ldm,
			workingVector,
			blockCache);
}

__device__ void applyOneHHVectD	(float* topElem,
				float* lowElem,
				int k,
				volatile float tau[],
				volatile float blockCache[])
{
	float alpha;
	__shared__ volatile float workV[32];

	/* Compute alpha = sum */
	if(TID == k)
		workV[TID] = *topElem;
	if(TID != k)
		workV[TID] = 0.0;

	workV[TID] += *lowElem * blockCache[TID + (k*32)];

	reduceSum(workV, 32);
	
	alpha = workV[0];

	/* Multiply by tau */
	alpha *= tau[k];

	/* Compute alpha *= a_tid,j*v_tid,k */
	if(TID == k)
		*topElem -= alpha;
			
	/* For lower element. */
	alpha *= blockCache[TID + (k*32)];
	*lowElem -= alpha;
}

__device__ void device_doQRDW	(float* blockA, float* blockB, float* blockTau,
				int ldm,
				volatile float tauVect[],
				volatile float blockCache[])//32*32 temp space
{
	/* Idea: for each column j of (AB)^T,
		apply HH vectors 0...j-1,
		compute new HH vector for column j,
		store essential part in blockCache 

	Uses one block cache, one vector storage. */
	int j, k, i;

	float 	topElem,
		lowElem,
		tau;

	for(j = 0; j < 32; j ++)
	{
		/* Apply previous HH vectors to column j. */
		if(TID <= j)
			topElem = blockA[TID + (j*ldm)];
		if(TID > j)
			topElem = 0;

		lowElem = blockB[TID + (j*ldm)];

		for(k = 0; k < j; k ++)
		{
			/* Compute b_tid,j = b_tid,j - tau*vv'*b_tid,j */
			applyOneHHVectD	(&topElem, &lowElem,
					k,
					tauVect,
					blockCache);
		}

		calcDoubleHHWY	(topElem,
				lowElem,
				j,
				blockCache);

		/* Compute new tau = 2/v'v */
		tau = 1.0;
		for(i = 0; i < 32; i ++)
			tau += blockCache[i + (j*32)] * blockCache[i + (j*32)];
		tau = 2.0/tau;
		
		if(TID == j)
			tauVect[j] = tau;

		/* Apply new vector to column. */
		applyOneHHVectD	(&topElem, &lowElem,
				j,
				tauVect,
				blockCache);

		/* Write back */
		if(TID <= j)
			blockA[TID + (j*ldm)] = topElem;
	}

	/* Write back lower block, containing householder Vectors. */
	for(j = 0; j < 32; j ++)
		blockB[TID + (j*ldm)] = blockCache[TID + (j*32)];
	
	blockTau[TID] = tauVect[TID];

	__threadfence();
}

__device__ void device_doQRD	(float* blockA,  float* blockB, float* blockTau,
				int ldm,
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
				blockTau,
				ldm,
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
	__threadfence();
}

__global__ void doQRD( float* blockA,  float* blockB, float* blockTau, int ldm)
{
	__shared__ volatile float workingVector[64];
	
	float rowA[32], rowB[32];

	device_doQRD	(blockA, blockB, blockTau, ldm,
			workingVector,
			rowA,
			rowB);
}

__device__ void device_MdoSAPP	(float* blockV,
				float* blockA,
				float* blockTau,
				int ldm,
				volatile float workingVect[],
				volatile float blockCache[])
{
	int 	j, k, i,
		tid = TID%32,
		group = TID/32;

	__shared__ volatile float tau[32];
	__shared__ volatile float groupCache[32*WARPS];
	
	volatile float *cacheCol = groupCache + (group*32);
	
	float 	alpha,
		belem;
	
	__syncthreads();
	/* Load tau Vector */
	if(TID < 32)
		tau[TID] = blockTau[TID];
	__syncthreads();
	
	/* Load Vectors */
	for(j = group; j < 32; j += WARPS)
	{
		if(tid < j)
		{
			blockCache[tid + (j*32)] = 0.0;
		}
		if(tid == j)
		{
			blockCache[tid + (j*32)] = 1.0;
		}
		if(tid > j)
		{
			blockCache[tid + (j*32)] = blockV[tid + (j*ldm)];
		}
	}
	__syncthreads();

	/* Compute b_j -= tau*v*v'b_j, for all vectors in blockCached V */
	for(j = group; j < 32; j += WARPS)
	{
		belem = blockA[tid + (j*ldm)];
		/* For each vector in block of vectors. */

		for(k = 0; k < 32; k ++)
		{
			/* Compute alpha = v'*b_j */
			cacheCol[tid] = blockCache[tid + (k*32)] * belem;

			if(tid < 16)
				cacheCol[tid] += cacheCol[tid+16];
			if(tid < 8)
				cacheCol[tid] += cacheCol[tid+8];
			//alpha = cacheCol[7];
			alpha = cacheCol[0];
			for(i = 1; i < 8; i ++)
				alpha += cacheCol[i];

			/* Compute alpha = tau * v_tid * alpha */
			alpha *= tau[k];
			alpha *= blockCache[tid + (k*32)];
			
			/* Compute belem -= alpha */
			belem -= alpha;
		}
		blockA[tid + (j*ldm)] = belem;
		__syncthreads();
	}
	__threadfence();
}
__device__ void device_doSAPP	(float* blockV,
				float* blockA,
				float* blockTau,
				int ldm,
				volatile float workingVector[],
				volatile float blockCache[])
{
	int 	j, k;

	__shared__ volatile float tau[32];
	
	float 	alpha,
		belem;
	
	/* Load tau Vector */
	tau[TID] = blockTau[TID];
	
	/* Load Vectors */
	//for(j = 0; j < 32; j ++) if(TID > j) blockCache[TID + (j*32)] = blockV[TID + (j*ldm)];

	for(j = 0; j < 32; j ++)
	{
		if(TID < j)
			blockCache[TID + (j*32)] = 0.0;
		if(TID == j)
			blockCache[TID + (j*32)] = 1.0;
	}
	
	/* Compute b_j -= tau*v*v'b_j, for all vectors in blockCached V */
	for(j = 0; j < 32; j ++)
	{
		belem = blockA[TID + (j*ldm)];
		/* For each vector in block of vectors. */
		for(k = 0; k < 32; k ++)
		{
			/* Compute alpha = v'*b_j */
			workingVector[TID] = blockCache[TID + (k*32)] * belem;
			reduceSum(workingVector, 32);
			alpha = workingVector[0];

			/* Compute alpha = tau * v_tid * alpha */
			alpha *= tau[k];
			alpha *= blockCache[TID + (k*32)];
			
			/* Compute belem -= alpha */
			belem -= alpha;
		}
		blockA[TID + (j*ldm)] = belem;
	}
	__threadfence();
}

__global__ void doSAPP	(float* blockV,
			float* blockA,
			float* blockTau,
			int ldm)
{
	__shared__ volatile float workingVector[32];
	__shared__ volatile float blockCache[32*32];
	
	device_MdoSAPP	(blockV, blockA, blockTau,
			ldm,
			workingVector,
			blockCache);
}

__device__ void applyOneHHVectDW(float* topElem,
				float* lowElem,
				int k,
				volatile float tau[],
				volatile float blockCache[],
				int group,//threadIdx.x/32
				int tid)//threadIdx.x%32
{
	float alpha;
	volatile float *col;
	int ind = tid + (k<<5);
	int i;

	__shared__ volatile float workV[WARPS*32];
	col = workV + (WARPS*group);

	col[tid] = *topElem;
	alpha = col[k];
	/* Compute alpha = sum */
	col[tid] = *lowElem * blockCache[ind];

	/* Partial reduction to find half of v'*b_j */
	/*if(tid < 16)
		col[tid] += col[tid + 16];
	if(tid < 8)
		col[tid] += col[tid + 8];*/

	/* Simple summation to find second half.
           Combining the two turned out to be faster than either on their own. */
	for(i = 0; i < 32; i ++)
		alpha += col[i];

	/* Multiply by tau */
	alpha *= tau[k];

	/* Compute alpha *= a_tid,j*v_tid,k */
	if(tid == k)
		*topElem -= alpha;
			
	/* For lower element. */
	alpha *= blockCache[ind];
	*lowElem -= alpha;
}

__device__ void device_MdoDAPP	(float* blockV,
				float* blockA,
				float* blockB,
				float* blockTau,
				int ldm,
				volatile float workingVector[],
				volatile float blockCache[])
{
	__shared__ volatile float tau[32];
	__shared__ volatile float workV[WARPS*32];

	float 	aelem, belem,
		alpha;


	int 	j, k, i,
		tid, group,
		refMat, refCache;
	
	tid = TID%32;
	group = TID/32;

	refMat = tid + group*ldm;
	refCache = tid + group*32;	
	__syncthreads();

	if(TID < 32)
		tau[TID] = blockTau[TID];

	__syncthreads();

	/* Load the essential HH vector block into shared cache. */
	for(j = group; j < 32; j += WARPS)
	{
		blockCache[refCache] = blockV[refMat];

		refMat += WARPS*ldm;
		refCache += WARPS*32;
	}

	__syncthreads();

	/* For each column of the result. */
	for(j = group; j < 32; j += WARPS)
	{
		//if(tid == 0)	printf("%d: column %d.\n", group, j);
		/* Load the elements of the column to process. */
		aelem = blockA[tid + (j*ldm)];
		belem = blockB[tid + (j*ldm)];
		__syncthreads();
		
		/* Compute and apply b_j = b_j - tau*vv'b_j
			for each Householder vector 1..32. */
		for(k = 0; k < 32; k ++)
		{
			/* Load the kth element into all threads. */
			workV[tid + (group*32)] = aelem;

			/* Store this as alpha. */
			alpha = workV[k + (group*32)];
	
			/* Load components of v'b_j to sum. */
			workV[tid + (group*32)] = belem * blockCache[tid + (k*32)];

			if(tid < 16)
				workV[tid + (group*32)] += workV[16 + tid + (group*32)];
			if(tid < 8)
				workV[tid + (group*32)] += workV[8 + tid + (group*32)];
			/* Compute v'b_j */
			for(i = 0; i < 8; i ++)
				alpha += workV[i + (group*32)];

			/* Multiply by kth tau. */
			alpha *= tau[k];
			
			/* Compute b_j -= alpha*v
				If kth thread in group, v is 1 at aelem. */
			if(tid == k)
				aelem -= alpha;

			/* Compute b_j -= alpha * v for lower half. */
			belem -= alpha * blockCache[tid + (k*32)];
			__syncthreads();
		}
		/* put the elements back. */
		blockA[tid + (j*ldm)] = aelem;
		blockB[tid + (j*ldm)] = belem;
		__syncthreads();
	}
	__threadfence();
}

__device__ void device_doDAPP	(float* blockV,
				float* blockA,
				float* blockB,
				float* blockTau,
				int ldm,
				volatile float workingVector[],
				volatile float blockCache[])
{
	__shared__ volatile float tau[32];

	float 	aelem, belem;

	int j, k, refMat, refCache;
	
	refMat = TID;
	refCache = TID;
	tau[TID] = blockTau[TID];

	/* Load the essential HH vector block into shared cache. */
	for(j = 0; j < 32; j ++)
	{
		blockCache[refCache] = blockV[refMat];

		refMat += ldm;
		refCache += 32;
	}

	/* For each column of the result. */
	for(j = 0; j < 32; j ++)
	{
		/* Load the elements of the vector to process. */
		aelem = blockA[TID + (j*ldm)];
		belem = blockB[TID + (j*ldm)];
		
		/* For each vector in blockV. */
		for(k = 0; k < 32; k ++)
		{
			/* Compute v'*b_j */
			applyOneHHVectD	(&aelem,
					&belem,
					k,
					tau,
					blockCache);
		}
		
		/* put the elements back. */
		blockA[TID + (j*ldm)] = aelem;
		blockB[TID + (j*ldm)] = belem;
	}
}

__global__ void doDAPP	(float* blockV,
			float* blockA,
			float* blockB,
			float* blockTau,
			int ldm)
{
	__shared__ volatile float workingVector[32];
	__shared__ volatile float blockCache[32*32];

	device_MdoDAPP	(blockV,
			blockA,
			blockB,
			blockTau,
			ldm,
			workingVector,
			blockCache);
}

/* Fetch and return a task's information from the task structure, at location
   given by ref. */
inline __device__ void retrieveTask( Task* ret, volatile Task* t )
{
	//__threadfence();
	int l;

	//read and return value
	//printf("tasktzp: %p: %d\n", &ret->l, ret->l);
	l = t->l;
	
	ret->l = l;//t->l;
	ret->m = t->m;
	ret->k = t->k;
	ret->taskStatus = DOING;
	ret->taskType = t->taskType;
}

__device__ int executeTask	(Task t,
				float* mat, float* matTau,
				int ldm, int n,
				volatile float workingVector[],
				volatile float blockCache[])
{
	float *blockV, *blockA, *blockB, *blockTau;

	//switch based on the type of task we've got
	//TIMER_TIC
	switch(t.taskType)
	{
		case QRS:
		{
			blockV = mat + CO(t.k*32,t.k*32,ldm);
			blockTau = matTau + CO(t.k*32,t.k*32,ldm);
			if(TID < 32)
			{
				TIMER_TIC
				device_doQRS	( blockV, blockTau, ldm, workingVector, blockCache );
				TIMER_TOC(tid_doingQRS);
			}
//			if(TID == 0)printf("%d: QRS at %d,%d\n", blockIdx.x, t.k, t.k);
			break;
		}
		case SAPP:
		{
			blockV = mat + CO(t.k*32,t.k*32,ldm);
			blockA = mat + CO(t.k*32,t.m*32,ldm);
			blockTau = matTau + CO(t.k*32,t.k*32,ldm);
			//if(TID < 32){
				//for(j = 0; j < 32; j ++) blockCache[TID + (j*32)] = 0;
				TIMER_TIC
				//device_MdoSAPP	( blockV, blockA, blockTau, ldm, workingVector, blockCache );
				TIMER_TOC(tid_doingSAPP);
			//}
//			if(TID == 0)printf("%d: SAPP from %d,%d to %d,%d\n", blockIdx.x, t.k, t.k, t.k, t.m);
			break;
		}
		case QRD:
		{
			blockA = mat + CO(t.k*32,t.k*32,ldm);
			blockB = mat + CO(t.l*32,t.k*32,ldm);
			blockTau = matTau + CO(t.l*32,t.k*32,ldm);
			if(TID < 32)
			{
				TIMER_TIC
				device_doQRDW	( blockA, blockB, blockTau, ldm, workingVector, blockCache );
				TIMER_TOC(tid_doingQRD);
			}
//			if(TID == 0)printf("%d: QRD on %d,%d; %d,%d\n", blockIdx.x, t.k, t.k, t.l, t.k);
			break;
		}
		case DAPP:
		{
			blockV = mat + CO(t.l*32,t.k*32,ldm);
			blockA = mat + CO(t.k*32,t.m*32,ldm);
			blockB = mat + CO(t.l*32,t.m*32,ldm);
			blockTau = matTau + CO(t.l*32,t.k*32,ldm);
			TIMER_TIC
			device_MdoDAPP	(blockV, blockA, blockB, blockTau, ldm,	workingVector, blockCache);
			TIMER_TOC(tid_doingDAPP);
//			if(TID == 0)printf("%d: DAPP from %d,%d to %d,%d; %d,%d\n", blockIdx.x, t.l, t.k, t.k, t.m, t.l, t.m);
			break;
		}
	}
	//TIMER_TOC(tid_working);
	__syncthreads();

	return 1;
}

__global__ void taskKernel	(float* matrix,
				float* matTau,
				int m, int n,
				int totTasks,
				volatile Task* taskGrid,
				int M, int N )
{
	TIMER_TIC
	__shared__ volatile float workVector[64];
	__shared__ volatile float blockCache[32*32];
	
	int taskid;
	__shared__ Task task;
	__shared__ int s_tid;

	/* repeat while there are still tasks undone */
	while(q.rec_count < totTasks )
	{
		/* retrieve task from the cuda queue */
		if(TID == 0)
		{
			TIMER_TIC
			taskid = runner_cuda_gettask();
			s_tid = taskid;
			TIMER_TOC(tid_gettingtasks)
		}

		/* have finished if taskid is less than 0. Might also have invalid task */
		if(s_tid < 0)
		{
			//if(q.rec_count < totTasks)asm("trap;");
			continue;
		}

		/* get the specifics of this task from the main task structure */
		if( TID == 0 )
		{
			TIMER_TIC
			task.l = taskGrid[taskid].l;
			task.m = taskGrid[taskid].m;
			task.k = taskGrid[taskid].k;
			task.taskType = taskGrid[taskid].taskType;
			//printf("%d: %d at (%d,%d)\n", blockIdx.x, task.taskType, task.l, task.m);
			TIMER_TOC(tid_taskinfo)
		}

		/* perform the activity specified by the task t */
		{TIMER_TIC
		executeTask	(task, matrix, matTau, m, n,
				workVector,
				blockCache);
		TIMER_TOC(tid_executing)}

		/* register task as finished in the task structure 
		At the same time, insert each newly activated task into the cuda queue */
		if( TID == 0 )
		{
			TIMER_TIC
			completeATask( taskGrid, M, N, task );
			TIMER_TOC(tid_completing)
		}
	}
	TIMER_TOC(tid_total);
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

int displayDeviceProps()
{
	struct cudaDeviceProp cu_p;
	cudaGetDeviceProperties(&cu_p, 0);

	/*printf("Printing device information:\n");
	printf("Device name: %s\n", cu_p.name);
	printf("Gmem: %2.4f GB\n", (float)cu_p.totalGlobalMem/1e9);
	printf("Smem per block: %5.2f KB\n", (float)cu_p.sharedMemPerBlock/1e3);
	printf("Regs per block: %d\n", cu_p.regsPerBlock);
	printf("Clock rate: %5.3f MHz\n", (float)cu_p.clockRate/1e3);
	printf("MP Count: %d\n", cu_p.multiProcessorCount);
	printf("Memory clock rate: %d KHz\n", cu_p.memoryClockRate);
	printf("Max threads per MP: %d\n", cu_p.maxThreadsPerMultiProcessor);*/
	
	return cu_p.clockRate;
}

extern "C"
void cudaQRTask(float* mat, int m, int n, int ldm, int maxblocks)
{
	int totalTasks, p = m/32, q = n/32, queuelen = p * q, j;
	volatile int *dev_data;
	//initialise task structure on GPU
	volatile Task* dev_taskGrid;

	enum cudaError cuerr;
	
	cudaEvent_t start, stop;

	float *dev_m, *dev_tau, time, clockS;
	float timers[ tid_count ];
	
	totalTasks = calcTotalTasks( p, q );

	clockS = displayDeviceProps();

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cuerr = cudaMalloc( (void**) &dev_taskGrid, p * q * sizeof(Task) );
	if(cuerr != cudaSuccess)
		printf("error allocating task grid\n");
	cuerr = cudaMalloc( (void**) &dev_data, queuelen * sizeof(int) );
	if(cuerr != cudaSuccess)
		printf("error allocating dev queue data\n");
	cuerr = cudaMalloc( (void**) &dev_m, m*n*sizeof(float) );
	if(cuerr != cudaSuccess)
		printf("error allocating dev mat\n");
	cuerr = cudaMalloc( (void**) &dev_tau, m*n*sizeof(float) );
	if(cuerr != cudaSuccess)
		printf("error allocating tau mat\n");

	for(j = 0; j < n; j ++)
	{
		cuerr = cudaMemcpy(dev_m + (j*m), mat + (j*ldm), m*sizeof(float), cudaMemcpyHostToDevice);
		if(cuerr != cudaSuccess)
			printf("error cpying dev mat\n");
	}

	cuerr = cudaMemcpyToSymbol( cuda_queue_size, &queuelen, sizeof(int), 0, cudaMemcpyHostToDevice );
	if(cuerr != cudaSuccess)
		printf("error cpying size\n");

	cudaEventRecord(start,0);
	/* initialise all structures for scheduling operations on the GPU */
	cuda_initScheduling<<<1,1>>>( 	dev_taskGrid, p, q,
					dev_data, queuelen, totalTasks );
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&time, start, stop);
	//printf("Set up time taken: %f\n", time);
	
	cudaEventRecord(start,0);

	//taskKernel<<<p*q > maxblocks ? maxblocks : p*q,32*WARPS>>>(	dev_m, dev_tau,
	taskKernel<<<p*q -1 > MXBLOCKS ? MXBLOCKS : p*q -1,32*WARPS>>>(	dev_m, dev_tau,
					m, n, totalTasks, dev_taskGrid, p, q );

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&time, start, stop);
	printf("Kernel time taken for %d: %f\n", p*q -1 > MXBLOCKS ? MXBLOCKS : p*q -1, time);
	//printf(": %f\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	cuerr = cudaMemcpyFromSymbol(timers, cuda_timers, tid_count*sizeof(float), 0, cudaMemcpyDeviceToHost );
	/*if( cuerr != cudaSuccess )
		printf("fail %d.\n", cuerr);
	else
		printf("Waiting for tasks: %5.3fms\nDoing operations: %5.3fms\nTotal time: %5.3fms\n", timers[tid_waitfortask]/128/clockS, timers[tid_working]/128/clockS, timers[ tid_total ]/128/clockS );

	for(j = tid_doingQRS; j <= tid_doingDAPP; j ++)
		printf("Doing thing %d: %5.3f\n", j, timers[j]/128/clockS);

	for(j = tid_gettingtasks; j <= tid_completing; j++)
		printf("Doing processing item %d: %5.3f\n", j, timers[j]/128/clockS);
	printf("Mutex time: %5.3f\n", timers[tid_mutex]/128/clockS);*/

	for(j = 0; j < n; j ++)
	{
		cuerr = cudaMemcpy(mat + (j*ldm), dev_m + (j*m), m*sizeof(float), cudaMemcpyDeviceToHost);
		if(cuerr != cudaSuccess)
			printf("copy back failed %d\n", cuerr);
	}

	cuerr = cudaFree(dev_m);
	if( cuerr != cudaSuccess)
		printf("error freeing m %d\n", cuerr);
	cuerr = cudaFree(dev_tau);
	if( cuerr != cudaSuccess)
		printf("error freeing tau %d\n", cuerr);
	cudaFree((Task *) dev_taskGrid);
	if( cuerr != cudaSuccess)
		printf("error freeing task grid %d\n", cuerr);
	cudaFree((int *) dev_data);
	if( cuerr != cudaSuccess)
		printf("error freeing matrix %d\n\n", cuerr);

	cudaDeviceReset();//printf("error freeing m %d\n", cuerr);
}
__global__ void kern_testDAPP	(float* mat,
				float* tau)
{
	/* blockV is block(1,0), blockA is (0, blockIdx.x), blockB is (1,blockIdx.x) */
	__shared__ volatile float workVector[64], block[32*32];

	__syncthreads();
	TIMER_TIC
	
	device_MdoDAPP	(mat + 32,
			mat + (blockIdx.x*64),
			mat + 32 + (blockIdx.x*64),
			tau,
			64,
			workVector, block);
	__syncthreads();
	//TIMER_TOC(tid_test);
}

extern "C"
void testDAPP(float* timings, int n, int nblocks)
{
	float 	*h_data, *dev_data,
		*h_tau, *dev_tau;
	float 	time, timers[tid_count];
	int i, b, t;

	cudaError_t cuerr;
	cudaEvent_t start, stop;

	for(t = 0; t < n; t ++)
	{
		srand(5);
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	
		h_data = (float*) malloc((1+nblocks)*64*32*sizeof(float));
		h_tau = (float*) malloc(32*sizeof(float));

		cudaMalloc((void**) &dev_data, (1+nblocks)*64*32*sizeof(float) );
		cudaMalloc((void**) &dev_tau, 32*sizeof(float));

		for(i = 0; i < 64*32; i ++)
			h_data[i] = ((float)(rand() % 101) - 50.0) / 50.0;

		for(i = 0; i < 32; i ++)
			h_tau[i] = ((float)(rand() % 101) - 50.0) / 50.0;

		for(b = 1; b <= nblocks; b ++)
		{
			for(i = 0; i < 64*32; i ++)
				h_data[i + (b*64*32)] = h_data[i];//use same data for all tests
		}

		cudaMemcpy(dev_data, h_data, (1+nblocks)*64*32*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_tau, h_tau, 32*sizeof(float), cudaMemcpyHostToDevice);

		//timers_reset<<<1,1>>>();

		cudaEventRecord(start,0);
		kern_testDAPP<<<nblocks,WARPS*32>>>(dev_data, dev_tau);
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);

		cudaMemcpyFromSymbol(timers, cuda_timers, tid_count*sizeof(float), 0, cudaMemcpyDeviceToHost );

		cudaEventElapsedTime(&time, start, stop);

		timings[t] = time * nblocks;
		//timings[t] = timers[tid_test];

		cuerr = cudaGetLastError();
		if(cuerr != cudaSuccess)
		{
			printf("ERROR!!! %d\n", cuerr);
			break;
		}
		cudaFree(dev_data);
		cudaFree(dev_tau);
	
		free(h_data);
		free(h_tau);
	
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		//cudaDeviceReset();
	}
}

void doCUDADAPP(float* mat)
{
	float* dev_mat;
	cudaError_t cuerr;
	
	cudaMalloc((void**) &dev_mat, 64*64*sizeof(float));
	
	cudaMemcpy(dev_mat, mat, 64*64*sizeof(float), cudaMemcpyHostToDevice);

	doDAPP<<<1,WARPS*32>>>(dev_mat + 32,		//V
			dev_mat	+ (32*64),	//A
			dev_mat + 32 + (32*64),	//B
			dev_mat,		//Tau
			64);			//ldm

	cudaMemcpy(mat, dev_mat, 64*64*sizeof(float), cudaMemcpyDeviceToHost);
	
	cuerr = cudaGetLastError();

	if(cuerr != cudaSuccess)
		printf("error %d\n", cuerr);

	cudaFree(dev_mat);
}

/*extern "C"
void cudaQRFull(float* mat, int m, int n)
{
	int i, j, k, p, q, s;
	int blockdm;

	float* dev_m, *dev_tau, *dev_K, *dev_V, *dev_A, *dev_B, *dev_T;

	cudaStream_t streams[NUMSTREAMS];
	
	for(k = 0; k < NUMSTREAMS; k ++)
		cudaStreamCreate(&streams[k]);

	p = m/32;
	q = n/32;

	blockdm = 32*m;//block to block dim along row

	cudaMalloc((void**) &dev_m, m*n*sizeof(float));
	cudaMalloc((void**) &dev_tau, m*n*sizeof(float));

	cudaMemcpy(dev_m, mat, m*n*sizeof(float), cudaMemcpyHostToDevice);

	dev_K = dev_m;
	dev_T = dev_tau;

	for(k = 0; k < q; k ++)
	{
		doQRS<<<1, 32, 0, streams[0]>>>(dev_K, dev_T, m);
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
			dev_T 

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
}*/
