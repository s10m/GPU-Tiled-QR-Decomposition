double* newMatrix(double*, int, int);
void deleteMatrix(double*);
void initMatrix(double*, int, int, int);
void printMatrix(double*, int, int);

double* multAB(double*, int, int, double*, int, int, int, int);

double** hhQR(double*, int, int, int, int, int, int);

double* calcvk(double*, int);
double do2norm(double*, int);
double* normalisev(double*, int, double);
void updateMatHHQRInPlace(double*, int, int, int, int, int, double*, int);
