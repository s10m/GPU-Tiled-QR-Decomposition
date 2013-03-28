#ifndef QRDECOMP_H
#define QRDECOMP_H
void* pthr_doTasks(void*);
void doPthrBcast(pthread_cond_t*, int*);

struct ThreadInfo
{
	float* mat, *wspace, *tau;
	int ldm, b;

	Task* taskGrid;
	int taskM, taskN;

	pthread_mutex_t *getTaskMutex, *getSigMutex;
	pthread_cond_t *newTasksCond;
	int *condMet;
};

void blockQR( int, int, int );

float* newMatrix(int, int);
void deleteMatrix(float*);
void initMatrix(float*, int, int, int);
void printMatrix(float*, int, int, int);
void copyMatrix(float*, int, int, float*);
int checkEqual(float*, float*, int, int, int);

void doATask(Task, float*, float*, int, int, float*);

float* multAB(float*, int, int, int, float*, int, int);

void qRSingleBlock(float*, int, int, int, float*);
void qRSingleBlock_WY(float*, float*, int, int, int, float*);
void qRDoubleBlock(float*, int, int, float*, int, int, float*);

void applySingleBlock(float*, int, int, int, float*);
void applyDoubleBlock(float*, int, float*, int, int, int, float*);

void insSingleHHVector(float*, int, float*);

void allocVectors(float***, int, int);

void calcvkSingle(float*, int, float*);
void calcvkDouble(float, int, float*, int, float*);

void updateSingleQ(float*, int, int, int, float*);
void updateSingleQInp(float*, int, int, int, float*);
void updateDoubleQZeros(float*, int, int, float*, int, int, float*, int);
void updateDoubleQ(float*, int, int, float*, int, int, float*);

float do2norm(float*, int);
#endif
