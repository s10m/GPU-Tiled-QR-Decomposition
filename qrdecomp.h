#ifndef QRDECOMP_H
#define QRDECOMP_H
void* pthr_doATask(void*);

struct doTaskInfo
{
	Task t;
	double* ptr, *taskVect;
	int b, ldm;
};

void blockQR(void);

double* newMatrix(int, int);
void deleteMatrix(double*);
void initMatrix(double*, int, int, int);
void printMatrix(double*, int, int, int);

void doATask(Task, double*, int, int, double*);

double* multAB(double*, int, int, int, double*, int, int);

void qRSingleBlock(double*, int, int, int, double*);
void qRDoubleBlock(double*, int, int, double*, int, int, double*);

void applySingleBlock(double*, int, int, int, double*);
void applyDoubleBlock(double*, int, double*, int, int, int, double*);

void insSingleHHVector(double*, int, double*);

void allocVectors(double***, int, int);

void calcvkSingle(double*, int, double*);
void calcvkDouble(double, int, double*, int, double*);

void updateSingleQ(double*, int, int, int, double*);
void updateSingleQInp(double*, int, int, int, double*);
void updateDoubleQZeros(double*, int, int, double*, int, int, double*, int);
void updateDoubleQ(double*, int, int, double*, int, int, double*);

double do2norm(double*, int);
#endif
