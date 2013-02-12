#ifndef GRIDSCHEDULER_H
#define GRIDSCHEDULER_H

#define tgrid(x,y) taskGrid[(((y)*M)+(x))]

#define TASK_AVAIL 0
#define TASK_NONE 1
#define TASK_DONE 2

enum Type {QRS, SAPP, QRD, DAPP};
enum Status {READY, DOING, DONE, NONE, NOTASKS};

typedef struct{
	enum Type taskType;
	int l, m, k;
	enum Status taskStatus;
} Task;

void doneATask(Task*, int, int, Task);
int getNextTask(Task*, Task*, int, int);
Task* initScheduler(int, int);

#endif
