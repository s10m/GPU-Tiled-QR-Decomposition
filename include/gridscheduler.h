#ifndef GRIDSCHEDULER_H
#define GRIDSCHEDULER_H

#define tgrid(x,y) taskGrid[(((y)*M)+(x))]

enum Type {QRS, SAPP, QRD, DAPP};
enum Status {READY, DOING, DONE, NONE};

typedef struct{
	enum Type taskType;
	int l, m, k;
	enum Status taskStatus;
} Task;

void doneATask(Task*, int, int, Task);
Task getNextTask(Task*, int, int);
Task* initScheduler(int, int);

#endif
