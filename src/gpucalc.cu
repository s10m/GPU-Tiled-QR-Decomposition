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
#define WARPS 1
/* WARPS MUST be factor of blocksize. */

enum Type {QRS, SAPP, QRD, DAPP};
enum Status {READY, DOING, DONE, NONE, NOTASKS};

typedef struct{
	enum Type taskType;
		/* The co-ordinates of the task in the task grid. */
	int 	l, m,
		/* The step at which the task was generated. */
		k;
	
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

/*
 * @brief Initialise the cuda_queue struct and sets the queue to all -1.
 *
 * @param qlen The number of elements in the queue.
 * @param totalNumTasks The total number of tasks to be computed.
 * @param newData Pointer to the queue.
 *
 * Should be run with launch configuration of <<<1,1>>>
 */
__device__ void init_cuda_queue	(int qlen,
				int totalNumTasks,
				volatile int *newData)
{
	int j;

	/* Initialise values in the struct. */
	q.first = 0;
	q.last = 0;
	q.rec_count = 0;
	q.count = totalNumTasks;
	q.data = newData;
	q.numIn = 0;

	/* Initialise the queue. */
	for( j = 0; j < qlen; j ++)
	{
		q.data[j] = -1;
	}
}

/*
 * @brief Initialises the task grid to default values, and inserts the initial task.
 *
 * @param taskGrid The matrix to be used to store the tasks.
 * @param M The number of rows in the task matrix.
 * @param N The number of columns in the task matrix.
 *
 * Runs through all indices and sets the values of the structs at that point to the defaults.
 * Should not be called until init_cuda_queue has returned.
 */
__device__ void init_cuda_scheduler	(volatile Task* taskGrid,
					int M, int N)
{
	int i , j, ref;

	/* Possibly loops the wrong way round. Might need to swap these.
	   Potential cause for not working with non-square. */
	for(j = 0; j < M; j ++)
	{
		ref = j*M;
		for(i = 0; i < N; i ++)
		{
			taskGrid[ref].l = i;
			taskGrid[ref].m = j;
			taskGrid[ref].taskStatus = NONE;
			taskGrid[ref].k = 0;
			taskGrid[ref].topBusy = 0;

			cuda_mutex_unlock(&taskGrid[ref].mutex);
			ref ++;
		}
	}

	/* Insert the initial task into the task grid. */
	makeTask( taskGrid, M, 0, 0, QRS, READY, 0 );
}

/*
 * @brief Changes the state of a task in the task structure and adds the index to the queue.
 * 
 * @param taskGrid The matrix of tasks.
 * @param M The number of rows in the task matrix.
 * @param l The row of the task to change the state of.
 * @param m The column to change the state of.
 * @param newType The type to change the task to.
 * @param newStatus The status to change the task to. Not necessarily required but potentially useful.
 * @param newK The value of k the new task should have.
 *
 * Checks if task has already been modified, locks task then changes task, setting the new status to READY.
 * Following this, the index is added to the cuda queue.
 */
__device__ void makeTask(volatile Task* taskGrid,
			int M,
			int l, int m,
			enum Type newType, enum Status newStatus,
			int newK )
{
	/* Lock the task. */
	cuda_mutex_lock(&tgrid(l,m).mutex);

	/* Check if it has been modified already. There are scenarios where this can happen. I think. */
	if( atomicCAS( (int *) &tgrid(l,m).taskStatus, (int) NONE, (int) READY) == NONE ||
		atomicCAS( (int *) &tgrid(l,m).taskStatus, (int) DONE, (int) READY ) == DONE )
	{
		/* Modify the struct at this location. */
		tgrid(l,m).taskType = newType;
		tgrid(l,m).k = newK;
	
		/* Add the index to the queue. */
		cuda_queue_puttask( TLOC(x,y) );
	}
	/* Unlock the task. */
	cuda_mutex_unlock(&tgrid(x,y).mutex);
}

/*
 * @brief Returns the type of a derived task, given a location and value of k.
          Does not need to check if location is in matrix.
 * @param p The row to use.
 * @param q The column to use.
 * @param k The value of k to use.
 * @returns The type of a derived task at (p,q) at time k.
 *
 * Works in O(1) time by comparing location and k.
 * The combination of these parameters uniquely determines the new type returned.
 */
__device__ enum Type getNextType(int p, int q,
				int k)
{
	enum Type ret;

	/* Note that (p,q) is never < (k,k) */
	
	/* If on same row as k, is either QRS or SAPP. */
	if(p == k)
	{
		/* If on diagonal (p,q) == (k,k), is QRS. */
		if(q == k)
			ret = QRS;
		/* If on same row, but on diagonal, is SAPP. */
		else if(q > k)
			ret = SAPP;
	}
	/* If on different row to k. */
	else if(p > k)
	{
		/* If on a different row, but same column i.e (p,q) == (p,k) is QRD. */
		if(q == k)
			ret = QRD;
		/* If anywhere else i.e (p,q) > (k,k), is DAPP. */
		else if(q > k)
			ret = DAPP;
	}

	return ret;
}

/* @brief Checks (-1,-1) < (l,m) < (M,N)
 * @param M The number of rows in the task matrix.
 * @param N The number of columns in the task matrix.
 * @param l The row to check.
 * @param m The column to check.
 * @returns 1 if true, 0 if false.
 */
__device__ int inGrid	(int M, int N,
			int l, int m)
{
	int ret = 1;

	/* Proceed by eliminating possibilities, modifying ret as soon as possible. */
	if (l >= M)
		ret = 0;
	else if (m >= N)
		ret = 0;
	else if (l < 0)
		ret = 0;
	else if (y < 0)
		ret = 0;

	return ret;
}

/* 
 * @brief Checks if the task at (x,y) has been completed.
 * @param taskGrid The task matrix.
 * @param M The number of rows in the matrix.
 * @param x The row to check.
 * @param y The column to check.
 * @param k The timestep to check.
 * @returns 1 if task has been done, 0 otherwise.
 *
 * If the task is at the current timestep, check if it has been completed.
 * If task's timestep is more then the checking timestep, the task has definitely been completed some time in the past.
 * If task's timestep is less than the checking timestep, the task has definitely not been completed.
 * Must be called with a mutex lock on (x,y).
 */
__device__ int genericdone	(volatile Task* taskGrid, int M,
				int x, int y, int k)
{
	int ret = 0;

	/* Check if at current timestep. */
	if(tgrid(x,y).k == k)
	{
		/* If it has been completed, ret is true. */
		if(tgrid(x,y).taskStatus == DONE)
			ret = 1;
	}
	/* If task is at later timestamp, has definitely been completed. */
	else if(tgrid(x,y).k > k)
		ret = 1;

	return ret;
}

/*
 * @brief Checks if a QRS has been done for timestep k.
 * @param taskGrid The task matrix.
 * @param M The number of rows in the task matrix.
 * @param k The timestep to check.
 * @returns 1 if QRS has been done, 0 otherwise.
 * 
 * Works by checking the (k,k)th task, if it has been done, and is a QRS.
 */
__device__ int qrsdone	(volatile Task* taskGrid, int M,
			int k)
{
	int ret = 0;
	
	/* Lock the task. */
	cuda_mutex_lock(&tgrid(k,k).mutex);
	/* Check if the task at (k,k) has been done. */
	if(genericdone(taskGrid, M, k, k, k))
	{
		/* Possibly redundant check. */
		if(tgrid(k,k).taskType == QRS)
			ret = 1;
	}
	/* Unlock the task. */
	cuda_mutex_unlock(&tgrid(k,k).mutex);

	return ret;
}

/* 
 * @brief Checks if a DAPP task at (x,y), k has been completed.
 * @param taskGrid The task matrix.
 * @param M The number of rows in the task matrix.
 * @param N The number of columns in the task matrix.
 * @param x The row to check.
 * @param y The column to check.
 * @param k The timestamp to check.
 * @returns 1 if has been done, 0 otherwise.
 */
__device__ int dappdone	(volatile Task* taskGrid, int M, int N,
			int x, int y, int k)
{
	int ret = 0;

	/* If requested is out of grid, return true. ??? */
	if(!inGrid(M, N, x, y))
		return 1;

	/* Lock the task. */
	cuda_mutex_lock(&tgrid(x,y).mutex);
	/* Check if the task there has been finished. */
	if(genericdone(taskGrid, M, x, y, k))
	{
		/* Check if finished task is DAPP. */
		if(tgrid(x,y).taskType == DAPP)
			ret = 1;
	}
	/* Unlock the task. */
	cuda_mutex_unlock(&tgrid(x,y).mutex);

	return ret;
}

/*
 * @brief Checks if a QRD at (x,y), k has been completed.
 * @param taskGrid The task matrix.
 * @param M The number of rows in the task matrix.
 * @param N The number of columns in the task matrix.
 * @param x The row to check.
 * @param y The column to check.
 * @param k The timestamp to check.
 * @returns 1 if has been done, 0 otherwise.
 */
__device__ int qrddone	(volatile Task* taskGrid, int M, int N,
			int x, int y, int k)
{
	int ret = 0;
	
	/* Lock the task. */
	cuda_mutex_lock(&tgrid(x,y).mutex);
	/* Check if task is finished. */
	if(genericdone(taskGrid, M, x, y, k))
	{
		/* Check if task there is a QRD. Possibly redundant. */
		if(tgrid(x,y).taskType == QRD)
			ret = 1;
	}
	/* Unlock task. */
	cuda_mutex_unlock(&tgrid(x,y).mutex);
	
	return ret;
}

/*
 * @brief Checks if it is possible to do a QRS at the location (x,y) == (k,k).
 * @param taskGrid The task matrix.
 * @param M The number of rows in the task matrix.
 * @param N The number of Columns in the task matrix.
 * @param x The row to check.
 * @param y The column to check.
 * @param k The timestamp to check.
 * @returns 1 if can do a QRS at (x,y), 0 otherwise.
 *
 * Must be called with (x,y) == (k,k).
 * Just checks if (x,y) is in matrix or not.
 */

__device__ int candoQRS	(volatile Task* taskGrid, int M, int N,
			int x, int y, int k)
{
	/* Return whether (x,y) is in the task matrix or not. */
	return inGrid(M, N, x, y);
}

/*
 * @brief Checks if it is possible to do a SAPP at the location (x,y), k.
 * @param taskGrid The task matrix.
 * @param M The number of rows in the task matrix.
 * @param N The number of Columns in the task matrix.
 * @param x The row to check.
 * @param y The column to check.
 * @param k The timestamp to check.
 * @returns 1 if can do SAPP at (x,y), k 0 otherwise.
 *
 * SAPP at (x,y), k requires the QRS at (k,k) to be completed.
 * Also required is whatever task at (x,y) for previous k (can't be further back than 1 timestep) to be completed.
 */
__device__ int candoSAPP(volatile Task* taskGrid, int M, int N,
			int x, int y, int k)
{
	int ret = 0;

	/* If on top row of grid, always return true. */
	if (!inGrid(M, N, x - 1, y))
		return 1;

	/* Check kth QRS is completed. */
	if(qrsdone(taskGrid, M, k))
	{
		/* Check the DAPP at (x,y), k-1 is completed. */
		if(dappdone(taskGrid, M, N, x, y, k - 1))
			ret = 1;
	}

	return ret;
}

/*
 * @brief Checks if it is possible to do a QRD at the location (x,y), k.
 * @param taskGrid The task matrix.
 * @param M The number of rows in the task matrix.
 * @param N The number of Columns in the task matrix.
 * @param x The row to check.
 * @param y The column to check.
 * @param k The timestamp to check.
 * @returns 1 if can do QRD at (x,y), k, 0 otherwise.
 *
 * QRD at (x,y), k requires the task at (x-1,y) to be completed.
 * If k > 0, requires the DAPP at (x,y), k-1 to be completed.
 */
__device__ int candoQRD(volatile Task* taskGrid, int M, int N, int x, int y, int k)
{
	int ret = 0;

	/* If (x,y) is not in the grid, return false. */
	if(!inGrid(M, N, x, y))
		return 0;

	/* Lock the task above. */
	cuda_mutex_lock(&tgrid(x-1,y).mutex);

	/* Check whatever is above is done. Is either QRS or QRD above, so generic check. */
	if(genericdone(taskGrid, M, x-1, y, k))
	{
		/* If first k, no need to check previous at (x,y) are done. return true. */
		if(k == 0)
			ret = 1;
		/* If k > 0, check if DAPP at (x,y), k-1 is done. If it is then return true, false otherwise. */
		else if(dappdone(taskGrid, M, N, x, y, k-1))
			ret = 1;
	}
	/* Unlock the above task. */
	cuda_mutex_unlock(&tgrid(x-1,y).mutex);

	return ret;
}

/*
 * @brief Checks if it is possible to do a DAPP at the location (x,y), k.
 * @param taskGrid The task matrix.
 * @param M The number of rows in the task matrix.
 * @param N The number of Columns in the task matrix.
 * @param x The row to check.
 * @param y The column to check.
 * @param k The timestamp to check.
 * @returns 1 if can do DAPP at (x,y), k, 0 otherwise.
 *
 * DAPP at (x,y), k requires both the SAPP/DAPP at (x-1,y), k completed,
 * AND the QRD at (x,k), k to be completed.
 * Also requires the task at (x,y), k-1 to be complete.
 */

__device__ int candoDAPP(volatile Task* taskGrid, int M, int N, int x, int y, int k)
{
	int ret = 0;
	/* Check if the QRD at (x,k) is done. */
	if(qrddone(taskGrid, M, N, x, k, k))
	{
		/* Lock the task above. */
		cuda_mutex_lock(&tgrid(x-1,y).mutex);
		
		/* Check SAPP/DAPP above is done. */
		if(genericdone(taskGrid, M, x-1, y, k))
		{
			/* If no previous tasks possible, return true. */
			if(k == 0)
				ret = 1;
			/* If k > 0, check DAPP at (x,y), k-1 is complete. */
			else if(dappdone(taskGrid, M, N, x, y, k-1))
					ret = 1;
		}
		/* Unlock the task above. */
		cuda_mutex_unlock(&tgrid(x-1,y).mutex);
	}

	return ret;
}

/* @brief Updates the task matrix given a completed task.
 * @param taskGrid The task matrix.
 * @param M The number of rows in the task matrix.
 * @param N The number of Columns in the task matrix.
 * @param t A struct containing details of the completed task.
 * 
 * First sets the old task to be DONE.
 * Then uses a switch statement to decide which new tasks to generate and places the ready indices into the queue.
 */

__device__ void completeATask	(volatile Task* taskGrid,
				int M, int N,
				Task t)
{
	/* Parameters of the incoming task. */
	int 	k, j, p, q;
	/* Types used for switches. */
	enum Type tType, tTypeNext;
	
	/* Read parameters from struct into variables. */
	p = t.l;
	q = t.m;
	k = tgrid(p,q).k;

	/* Get the type to be used in the switch. */
	tType = getNextType(p, q, k);
	
	/* Lock the finished task. */
	cuda_mutex_lock(&tgrid(p,q).mutex);
	/* Register the finished task as done. */
	tgrid(p,q).taskStatus = DONE;
	/* Unlock the finished task. */
	cuda_mutex_unlock(&tgrid(p,q).mutex);
	
	/* Main switch. Decides based on type of finished task what to update and where. */
	switch(tType)
	{
		/* QRS at (p,q), k unlocks:
		   - QRD at (p+1,q), k
		   - Row of SAPP at (p,(q+1):N), k */
		case QRS:
		{
			if(candoQRD(taskGrid, M, N, p+1, q, k))
				makeTask(taskGrid, M, p+1, q, QRD, READY, k);

			for(j = k + 1; j < N; j ++)
			{
				if(candoSAPP(taskGrid, M, N, p, j, k))
					makeTask(taskGrid, M, p, j, SAPP, READY, k);
			}

			break;
		}
		/* SAPP at (p,q), k unlocks:
		   - DAPP at (p+1,q), k */
		case SAPP:
		{
			if(candoDAPP(taskGrid, M, N, p+1, q, k))
				makeTask(taskGrid, M, p+1, q, DAPP, READY, k);

			break;
		}
		/* QRD at (p,q), k unlocks:
		   - QRD at (p+1,q), k
		   - Row of DAPP at (p,(q+1):N), k */
		case QRD:
		{
			if(candoQRD(taskGrid, M, N, p+1, q, k))
				makeTask(taskGrid, M, p+1, q, QRD, READY, k);

			for(j = k + 1; j < N; j ++)
			{
				if(candoDAPP(taskGrid, M, N, p, j, k))
					makeTask(taskGrid, M, p, j, DAPP, READY, k);
			}
			
			break;
		}
		/* DAPP at (p,q), k unlocks:
		   - different tasks depending on type of (p,q) at k+1
		   - DAPP at (p+1,q), k */
		case DAPP:
		{
			/* Get type for (p,q) at next timestep. */
			tTypeNext = getNextType(p, q, k + 1);

			/* Switch on this new "next" type. */
			switch(tTypeNext)
			{
				/* DAPP at (k+1,k+1), k+1 unlocks:
				   - QRS at (k+1,k+1), k+1 */
				case QRS:
				{
					if(candoQRS(taskGrid, M, N, p, q, k + 1))
						makeTask(taskGrid, M, p, q, QRS, READY, k + 1);

					break;
				}
				/* DAPP at (k+1,q), k+1 unlocks:
				   - SAPP at (k+1,q), k+1 */
				case SAPP:
				{
					if(candoSAPP(taskGrid, M, N, p, q, k + 1))
						makeTask(taskGrid, M, p, q, SAPP, READY, k + 1);

					break;
				}
				/* DAPP at (p,k+1), k+1 unlocks:
				   - QRD at (p,k+1), k+1 */
				case QRD:
				{
					if(candoQRD(taskGrid, M, N, p, q, k + 1))
						makeTask(taskGrid, M, p, q, QRD, READY, k + 1);

					break;
				}
				/* DAPP at (p,q) > (k+1,k+1) unlocks:
				   - DAPP at (p,q), k+1 */
				case DAPP:
				{
					if(candoDAPP(taskGrid, M, N, p, q, k + 1))
						makeTask(taskGrid, M, p, q, DAPP, READY, k + 1);

					break;
				}
			}

			if(candoDAPP(taskGrid, M, N, p + 1, q, k))
				makeTask(taskGrid, M, p + 1, q, DAPP, READY, k);
			
			break;
		}
	}
}

/*
 * @brief Performs a reduction sum on a vector, storing the result in v[0].
 * @param sumVector SMEM The array to compute over.
 * @param startN The number of elements in the vector.
 *
 * Computes a binary reduction to find the sum of the elements in the vector,
 * storing the result in the first element of the vector.
 * should be called where threadDim.x <= 1 warp
 */
__device__ void reduceSum	(volatile float* sumVector,
				char startN)
{
	/* Start with n = startN/2 */
	char n = startN >> 1;

	while(n > 0)
	{
		/* Use first n threads to add up to 2n elements. */
		if(TID < n)	sumVector[TID] = sumVector[TID] + sumVector[TID + n];
		
		/* Divide n by 2 for next iteration. */
		n = n >> 1;
	}
}	

/*
 * @brief Computes a householder vector for a single tile QR decomposition, storing the result in a shared array.
 * @param matelem The TIDth element of the column to be used for calculation.
 * @param hhVector SMEM The array used to calculate and store the vector.
 * @param k The timestep at which this is called. i.e where the diagonal falls.
 * 
 * Calculates a householder vector for the vector passed as elements into a shared array.
 * Used only by SGEQRF_CUDA.
 */
__device__ void calcHH	(float matelem,
			volatile float hhVector[],
			int k)
{
	/* To store the value of 1/(v[k] + sign(v[k])*||v||). */
	float localdiv;
	/* Stores the sign of v[k] */
	int sign;

	/* Read vectors into shared memory from below the diagonal. */
	if(TID >= k)
		hhVector[TID] = matelem;

	/* Above the diagonal, set to 0. */
	if(TID < k)
		hhVector[TID] = 0;

	/* Square each element to find ||v|| = sqrt(sum(v[i]^2)) */
	hhVector[TID] *= hhVector[TID];
	
	/* Reduce to find sum of squares. */
	reduceSum(hhVector, 32);

	/* Set localdiv to equal ||v|| */
	localdiv = sqrt(hhVector[0]);

	/* According to Householder algorithm in Golub & Van Loan. */
	if(localdiv != 0.0)
	{
		/* Load shared to communicate v[k] to all threads. */
		hhVector[TID] = matelem;

		/* Calculate the sign of v[k] locally. */
		sign = hhVector[k] >= 0 ? 1 : -1;
		
		/* Compute and store localdiv = (||v||)*sign(v[k]) */
		localdiv *= sign;

		/* Compute and store localdiv = v[k] + (||v||*sign(v[k])) */
		localdiv += hhVector[k];

		/* Finally compute and store localdiv = 1/(v[k] + ||v||*sign(v[k])) */
		localdiv = 1.0f/localdiv;
	}
	/* If norm is 0, do not change v. */
	else
		localdiv = 1.0f;

	/* Compute and store in shared memory v = v/(v[k] + ||v||*sign(v[k])) */
	if(TID < k)
		hhVector[TID] = 0.0f;
	/* v[k]/v[k] = 1 */
	if(TID == k)
		hhVector[TID] = 1.0f;
	if(TID > k)
		hhVector[TID] = matelem * localdiv;
}

/*
 * @brief Applies a householder vector to the submatrix starting at (k,k), also inserts the kth tau value into tau[k].
 * @param blockCache SMEM The 1024-element array storing the full tile to which the vector will be applied.
 * @param matTau The tau vector.
 * @param workingVector SMEM The array used in the computation as a scratchpad.
 * @param k Defines the submatrix.
 * @param hhVectorelem Element of the cumulatively-stored householder vector to apply.
 *
 * Works by first calculating the kth tau,
 * then proceeding to apply the vector to successive columns of the matrix A_j
 * with the formula A_j = A_j - tau[k]*v*(v'A_j).
 * The kth tau is then stored into memory.
 */
__device__ void applyHH(volatile float blockCache[],
			float* matTau,
			volatile float workingVector[],
			int k,
			float hhVectorelem)
{
	float 	ktau,
		/* Stores the value of ktau*v*v'*A_j */
		z;

	int 	j;

	/* Read householder vector into shared memory to calculate ktau = 2/v'v */
	workingVector[TID] = hhVectorelem;
	/* Square the elements to start finding v'v */
	workingVector[TID] *= workingVector[TID];

	/* Perform a reduction to compute v'v and store the result in v[0]. */
	reduceSum(workingVector, 32);

	/* Finally compute ktau = 2/v'v */
	ktau = 2.0 / workingVector[0];
	
	/* For the columns of the submatrix starting at (k,k). */
	for(j = k; j < 32; j ++)
	{
		/* Fill the shared memory to calculate v'*A_j */
		workingVector[TID] = blockCache[TID+(j*32)] * hhVectorelem;

		/* Perform reduction to compute v'*A_j, storing the result in v[0] */
		reduceSum(workingVector, 32);
		
		/* Compute and store z = ktau*v[tid] */
		z = hhVectorelem * ktau;

		/* Compute and store z = (v'A_j)*(ktau*v[tid]) */
		z *= workingVector[0];
		
		/* Apply with A_j[tid] = A_j[tid] - v'A_j*ktau*v[tid] */
		blockCache[TID + (j*32)] -= z;
	}
	
	/* Store the essential (below diagonal) portion of householder vector below the diagonal in the block at column k. */
	if(TID > k)
		blockCache[TID + k*32] = hhVectorelem;
	
	/* Store the kth tau. */
	if(TID == 0)
		matTau[k] = ktau;
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
	sign = first >= 0.f ? 1.f : -1.f;

	if(alpha != 0.0f)
	{
		/* Zeroth element is = first + sign*norm */
		alpha = first + sign * alpha;
		alpha = 1.0f/alpha;
	}
	else
		alpha = 1.0f;

	//topElem *= alpha;
	lowElem *= alpha;
	
	blockCache[TID + (k*32)] = lowElem;
}

__device__ void SGEQRF_CUDA	( float* matrix, float* matTau,
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
	
	SGEQRF_CUDA	(matrix, tau,
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
		workV[TID] = 0.0f;

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

__device__ void STSQRF_CUDA	(float* blockA, float* blockB, float* blockTau,
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
			topElem = 0.f;

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
		tau = 1.0f;
		for(i = 0; i < 32; i ++)
			tau += blockCache[i + (j*32)] * blockCache[i + (j*32)];
		tau = 2.0f/tau;
		
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

__device__ void SLARFT_CUDA	(float* blockV,
				float* blockA,
				float* blockTau,
				int ldm,
				volatile float tau[],
				volatile float blockCache[])
{
	int 	j, k, i,
		tid = TID%32,
		group = TID/32;

	__shared__ volatile float groupCache[32*WARPS];
	
	volatile float *cacheCol = groupCache + (group*32);
	
	float 	alpha,
		belem;
	
	__syncthreads();
	/* Load tau Vector */
	if(TID < 32)
		tau[TID] = blockTau[TID];
	
	/* Load Vectors */
	for(j = group; j < 32; j += WARPS)
	{
		if(tid < j)
		{
			blockCache[tid + (j*32)] = 0.0f;
		}
		if(tid == j)
		{
			blockCache[tid + (j*32)] = 1.0f;
		}
		if(tid > j)
		{
			blockCache[tid + (j*32)] = blockV[tid + (j*ldm)];
		}
		__syncthreads();
	}
	__syncthreads();

	/* Compute b_j -= tau*v*v'b_j, for all vectors in blockCached V */
	for(j = group; j < 32; j += WARPS)
	{
		//if(group == 0)
		//{
		belem = blockA[tid + (j*ldm)];
		//__syncthreads();
		/* For each vector in block of vectors. */

		for(k = 0; k < 32; k ++)
		{
			/* Compute alpha = v'*b_j */
			cacheCol[tid] = blockCache[tid + (k*32)] * belem;
		//	__syncthreads();

			if(tid < 16)
				cacheCol[tid] += cacheCol[tid+16];
		//	__syncthreads();
			if(tid < 8)
				cacheCol[tid] += cacheCol[tid+8];
		//	__syncthreads();

			alpha = cacheCol[0];
			for(i = 1; i < 8; i ++)
				alpha += cacheCol[i];

			/* Compute alpha = tau * v_tid * alpha */
			alpha *= tau[k];
			alpha *= blockCache[tid + (k*32)];
			
			/* Compute belem -= alpha */
			belem -= alpha;
			__syncthreads();
		}
		blockA[tid + (j*ldm)] = belem;//}
		__syncthreads();
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
	
	SLARFT_CUDA	(blockV, blockA, blockTau,
			ldm,
			workingVector,
			blockCache);
}

__device__ void SSSRFT_CUDA	(float* blockV,
				float* blockA,
				float* blockB,
				float* blockTau,
				int ldm,
				volatile float tau[],
				volatile float blockCache[])
{
	__shared__ volatile float workV[WARPS*32];
	__syncthreads();

	float 	aelem, belem,
		alpha;


	int 	j, k, i,
		tid, group,
		refMat, refCache;
	
	tid = TID%32;
	group = TID/32;

	refMat = tid + group*ldm;
	refCache = tid + group*32;	

	if(TID < 32)
		tau[TID] = blockTau[TID];
	__syncthreads();


	/* Load the essential HH vector block into shared cache. */
	for(j = group; j < 32; j += WARPS)
	{
		blockCache[refCache] = blockV[refMat];

		refMat += WARPS*ldm;
		refCache += WARPS*32;
		__syncthreads();
	}

	__syncthreads();

	/* For each column of the result. */
	for(j = group; j < 32; j += WARPS)
	{
		//if(tid == 0)	printf("%d: column %d.\n", group, j);
		/* Load the elements of the column to process. */
		aelem = blockA[tid + (j*ldm)];
		belem = blockB[tid + (j*ldm)];
		
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
			if(tid == k)	aelem -= alpha;

			/* Compute b_j -= alpha * v for lower half. */
			belem -= alpha * blockCache[tid + (k*32)];
			__syncthreads();
		}
		/* put the elements back. */
		blockA[tid + (j*ldm)] = aelem;
		blockB[tid + (j*ldm)] = belem;
		__syncthreads();
	}
	__syncthreads();
	__threadfence();
}

__global__ void doDAPP	(float* blockV,
			float* blockA,
			float* blockB,
			float* blockTau,
			int ldm)
{
	__shared__ volatile float workingVector[32];
	__shared__ volatile float blockCache[32*32];

	SSSRFT_CUDA	(blockV,
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
	__syncthreads();
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
				SGEQRF_CUDA	( blockV, blockTau, ldm, workingVector, blockCache );
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
				TIMER_TIC
				SLARFT_CUDA	( blockV, blockA, blockTau, ldm, workingVector, blockCache );
				TIMER_TOC(tid_doingSAPP);
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
				STSQRF_CUDA	( blockA, blockB, blockTau, ldm, workingVector, blockCache );
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
			SSSRFT_CUDA	(blockV, blockA, blockB, blockTau, ldm,	workingVector, blockCache);
			TIMER_TOC(tid_doingDAPP);
//			if(TID == 0)printf("%d: DAPP from %d,%d to %d,%d; %d,%d\n", blockIdx.x, t.l, t.k, t.k, t.m, t.l, t.m);
			break;
		}
	}
	//TIMER_TOC(tid_working);
	__threadfence();

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
		__syncthreads();

		/* have finished if taskid is less than 0. Might also have invalid task */
		if(s_tid < 0)
		{
			//if(q.rec_count < totTasks)asm("trap;");
			//continue;
			return;
		}

		__syncthreads();
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
		__syncthreads();

		/* perform the activity specified by the task t */
		{TIMER_TIC
		executeTask	(task, matrix, matTau, m, n,
				workVector,
				blockCache);
		TIMER_TOC(tid_executing)}
		__syncthreads();

		/* register task as finished in the task structure 
		At the same time, insert each newly activated task into the cuda queue */
		if( TID == 0 )
		{
			TIMER_TIC
			completeATask( taskGrid, M, N, task );
			TIMER_TOC(tid_completing)
		}
		__syncthreads();
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

	printf("(%d,%d)\n%d tasks\n",m*32, n*32, ret);
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

	//taskKernel<<<1,32*WARPS>>>(	dev_m, dev_tau,
	taskKernel<<<p*q > maxblocks ? maxblocks : p*q,32*WARPS>>>(	dev_m, dev_tau,
	//taskKernel<<<p*q > MXBLOCKS ? MXBLOCKS : p*q,32*WARPS>>>(	dev_m, dev_tau,
					m, n, totalTasks, dev_taskGrid, p, q );

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&time, start, stop);
	//printf("Kernel time taken for %d: %f\n", p*q > maxblocks ? maxblocks : p*q, time);//p*q/2 > MXBLOCKS ? MXBLOCKS : p*q/2, time);
	printf("GPU: %5.3f ms\n", time);
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
	
	SSSRFT_CUDA	(mat + 32,
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
