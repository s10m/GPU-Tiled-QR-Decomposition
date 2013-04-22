CUDA_FIX_MEMISSUES=-dlcm=cg
CUDA_MYFLAGS=-O3 --maxrregcount=24 -lineinfo -src-in-ptx -Xptxas="-v $(CUDA_FIX_MEMISSUES)" -arch=sm_20 -use_fast_math
qr : obj/qrdecomp.o obj/gridscheduler.o obj/gpucalc.cu.o makefile
	nvcc $(CUDA_MYFLAGS) obj/gpucalc.cu.o obj/qrdecomp.o obj/gridscheduler.o -lm -lpthread -o qr
qr-nocuda : obj/qrdecomp.o obj/gridscheduler.o makefile
	gcc -O3 obj/qrdecomp.o obj/gridscheduler.o -lm -lpthread -o qr-nocuda
obj/gpucalc.cu.o : include/gpucalc.h src/gpucalc.cu makefile
	nvcc $(CUDA_MYFLAGS) -c src/gpucalc.cu -o obj/gpucalc.cu.o
obj/qrdecomp.o : qrdecomp.c qrdecomp.h
	gcc -c -Wunused qrdecomp.c -g -o obj/qrdecomp.o
obj/gridscheduler.o : include/gridscheduler.h src/gridscheduler.c
	gcc -c src/gridscheduler.c -g -Wall -Wunused -o obj/gridscheduler.o
clean :
	rm -f obj/* qr
remake :
	make clean
	make
