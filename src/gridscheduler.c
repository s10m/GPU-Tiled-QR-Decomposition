#include <stdio.h>
#include <stdlib.h>

#include "../include/gridscheduler.h"

void makeTask(Task* taskGrid, int M, int x, int y, enum Type newType, enum Status newStatus, int newK)
{
	tgrid(x,y).taskType = newType;
	tgrid(x,y).taskStatus = newStatus;
	tgrid(x,y).k = newK;
}

enum Type getNextType(int p, int q, int k)
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

int inGrid(int M, int N, int x, int y)//1 if (x,y) in grid, 0 if not
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
int genericdone(Task* taskGrid, int M, int x, int y, int k)
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
int qrsdone(Task* taskGrid, int M, int k)
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
int dappdone(Task* taskGrid, int M, int N, int x, int y, int k)
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

//checks if double qr has been performed on (x,y) at step k
int qrddone(Task* taskGrid, int M, int N, int x, int y, int k)
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
int candoQRS(Task* taskGrid, int M, int N, int x, int y, int k)
{
	return inGrid(M, N, x, y);
}

//if can apply at (x,y) step k, returns 1. 0 otherwise
int candoSAPP(Task* taskGrid, int M, int N, int x, int y, int k)
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

int candoQRD(Task* taskGrid, int M, int N, int x, int y, int k)
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

int candoDAPP(Task* taskGrid, int M, int N, int x, int y, int k)
{
	int ret = 0;
	//checkqrd(x,k)k done
	if(qrddone(taskGrid, M, N, x, k, k))
	{
		//checkgendone(x-1,y)k done
		if(genericdone(taskGrid, M, x-1, y, k))
		{
			//checkdapp(x,y)k-1 done6
			if(k == 0)
				ret = 1;
			else if(dappdone(taskGrid, M, N, x, y, k-1))
				ret = 1;
		}
	}

	return ret;
}

void doneATask(Task* taskGrid, int M, int N, Task t)
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
			for(j = k + 1; j < N; j ++)//check along row
			{
				if(candoSAPP(taskGrid, M, N, p, j, k))
					makeTask(taskGrid, M, p, j, SAPP, READY, k);
			}

			if(candoQRD(taskGrid, M, N, p+1, q, k))//check one below
				makeTask(taskGrid, M, p+1, q, QRD, READY, k);
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
			for(j = k + 1; j < N; j ++)
			{
				if(candoDAPP(taskGrid, M, N, p, j, k))//check along row
					makeTask(taskGrid, M, p, j, DAPP, READY, k);
			}
			
			if(candoQRD(taskGrid, M, N, p+1, q, k))//check one below
				makeTask(taskGrid, M, p+1, q, QRD, READY, k);
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

//returns 0 if success, 1 if tasks in progress, 2 if complete
int getNextTask(Task *t, Task* taskGrid, int M, int N)
{
	int i, j, r = TASK_DONE;
	Task ret;
	ret.taskStatus = NONE;

	/**
	 * THIS COULD GET VERY COMPLICATED,
	 * but for now it's just the first ready task.
	 */

	for(i = M-1; i > -1; i --)
	{
		for(j = N-1; j > -1; j --)
		{
			switch(tgrid(i,j).taskStatus)
			{
				case READY:
				{
					tgrid(i,j).taskStatus = DOING;
					ret = tgrid(i,j);
					r = TASK_AVAIL;
					break;
				}			
				case DOING:
				{
					r = TASK_NONE;
					break;
				}
				default:{}
					
			}
			if(ret.taskStatus == DOING)
				break;
		}
		if(ret.taskStatus == DOING)//if found
			break;
	}

	*t = ret;
	return r;
}

Task* initScheduler(int M, int N)
{
	Task* taskGrid;
	int i, j;

	taskGrid = malloc(M*N*sizeof(Task));

	for(i = 0; i < M; i ++)
	{
		for(j = 0; j < N; j ++)
		{
			tgrid(i,j).l = i;
			tgrid(i,j).m = j;
			tgrid(i,j).taskStatus = NONE;
			tgrid(i,j).k = 0;
		}
	}

	makeTask(taskGrid, M, 0, 0, QRS, READY, 0);

	return taskGrid;
}

/*int main(int argc, char* argv[])
{
	Task* taskGrid, ntask;
	int N = 3, M = N + 1, i, j;

	taskGrid = initScheduler(M, N);

	for(j = 0; j < 22; j ++)
	{
		ntask = getNextTask(taskGrid, M, N);
		if(ntask.taskStatus == NONE)
		{
			printf("done here.\n_____________________\n");
			break;
		}
		printf("(%d,%d) ", ntask.l, ntask.m);
		printf("t = %d k = %d\n", ntask.taskType, ntask.k);
		doneATask(taskGrid, M, N, ntask);
	}

	for(i = 0; i < M; i ++)
	{
		for(j = 0; j < N; j ++)
			printf("(%d,%d) %d %d %d\n", i, j, tgrid(i,j).taskType, tgrid(i,j).k, tgrid(i,j).taskStatus);
	}

	return 0;
}*/
